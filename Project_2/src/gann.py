import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
import tflowtools as TFT
import math

""" This is the GANN from tutor3.py, but improved/modified for this project. """


class Gann():
    grabVarPlotType = 'hinton' # 'hinton' or 'matrix'

    def __init__(self, name, netDims, cMan, hiddenActivationFunc = 'relu', outputActivationFunc = 'softmax',
                 lossFunc = 'MSE', optimizer = 'gradient_descent', optimizerParams = None, learningRate = 0.1, momentum = 0.1, weightRange = (-1,1), weightInitType = 'normalized',
                 mbs = 10):

        # SCENARIO PARAMETERS
        self.networkDims = netDims                        # Sizes of each layer of neurons
        self.hiddenActivationFunc = hiddenActivationFunc  # Activation function to use for each hidden layer
        self.outputActivationFunc = outputActivationFunc  # Activation function for the output of the network
        self.lossFunc = lossFunc                          # Quantity to minimize during training
        self.optimizer = optimizer                        # Optimizer used in learning to minimize the loss. 'gradient_descent' or 'momentum'
        self.optimizerParams = optimizerParams
        self.learningRate = learningRate                  # How large steps to take in the direction of the gradient
        self.momentum = momentum                          # The momentum, only relevant when self.optimizer = 'momentum'
        self.weightInit = weightRange                     # Upper and lower band for random initialization of weights
        self.weightInitType = weightInitType
        self.caseMan = cMan                               # Case manager object with a data source
        self.miniBatchSize = mbs                          # Amount of cases in each batch which are used for each run
        self.grabVars = []                                # Variables (weights and/or biases) to be monitored (by gann code) during a run.
        self.mapVars = []
        self.dendrogramVars = []

        # Grabbed variables ( defined in the call to run() )
        self.displayWeights = []                          # List of the weight arrays(their hidden layer indices) to be visualized at the end of the run. 0 is the first hidden layer
        self.displayBiases = []                           # List of the bias vectors(their hidden layer indices) to be visualized at the end of the run. 0 is the first hidden layer
        # Mapping variables (defined when a call to run_mapping() is made)
        self.mapBatchSize = 0                             # Size of batch of cases used for a map test. 0 indicates no map test
        self.mapLayers = []                               # List of layers(their indices) to be visualized during a map test. 0 is the input layer
        self.mapDendrograms = []                          # List of layers(their indices) whose activation patterns will be used to make dendrograms. 0 is the first hidden layer

        # CONVENIENCE PARAMETERS
        self.name = name                                  # Frequency of running validation runs
        self.globalTrainingStep = 0                       # Enables coherent data-storage during extra training runs (see runmore)
        self.input = None                                 # Pointer to the input of the network, where to feed the network cases/mbs
        self.target = None                                # Correct classification for each incoming case
        self.output = None                                # Pointer to the (softmaxed)output of the network
        self.rawOutput = None
        self.probes = None                                # Pointer to the probes (biases/weights monitored by tensorboard) which TF keeps track of
        self.error = None                                 # TF loss function variable created by specifying lossFunc
        self.output = None
        self.trainer = None


        # DATA STORAGE
        self.grabvarFigures = []                          # One matplotlib figure for each grabvar
        self.validationHistory = []
        self.error_history = []
        self.modules = []
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabVars.append(self.modules[module_index].getvar(type))
        self.grabvarFigures.append(PLT.figure())


    # Assumed that layer_index = 0 means that we want to monitor the input to the first hidden layer.
    # layer index = 1 means that we want to monitor the output of the first hidden layer, i-e the input of the second hidden layer
    def add_mapvars(self):
        for layer_index in self.mapLayers:
            if layer_index == (len(self.modules)): # the last output
                self.mapVars.append(self.modules[layer_index - 1].getvar('out'))
            else:
                self.mapVars.append(self.modules[layer_index].getvar('in'))

    def add_dendrogramvars(self):
        for layer_index in self.mapDendrograms:
            if (layer_index == (len(self.modules) - 1)) and self.outputActivationFunc == 'softmax':
                self.dendrogramVars.append(self.rawOutput)
            else:
                self.dendrogramVars.append(self.modules[layer_index].getvar('out'))

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self,module): self.modules.append(module)

    def build(self):
        tf.reset_default_graph()
        num_inputs = self.networkDims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all modules and connect them
        for i,outsize in enumerate(self.networkDims[1:]):
            if i == (len(self.networkDims)-2):
                gmod = Gannmodule(self, i, invar, insize, outsize, self.outputActivationFunc, initWeightRange=self.weightInit)
                self.output = gmod.output
                continue
            gmod = Gannmodule(self, i, invar, insize, outsize, self.hiddenActivationFunc, initWeightRange = self.weightInit)
            invar = gmod.output
            insize = gmod.outsize
        self.target = tf.placeholder(tf.float64, shape=(None,gmod.outsize), name='Target')
        self.configure_learning()



    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        if self.lossFunc == 'MSE':
            self.error = tf.losses.mean_squared_error(self.target, self.output)
            #self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        elif self.lossFunc == 'softmax_cross_entropy':
            self.error = tf.losses.softmax_cross_entropy(self.target, self.rawOutput)
        elif self.lossFunc == 'sigmoid_cross_entropy':
            self.error = tf.losses.sigmoid_cross_entropy(self.target, self.rawOutput)
        else:
            raise AssertionError("Unknown loss function: " + self.lossFunc)
        #self.output = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        if self.optimizer == 'gradient_descent':
            optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.learningRate, epsilon = self.optimizerParams[0])
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(self.learningRate)
        elif self.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.learningRate, momentum = self.optimizerParams[0], use_nesterov = True)
        else:
            raise AssertionError("Unknown optimizer: " + self.optimizer)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, epochs=100, continued=False):
        if not(continued):
            self.error_history = []
            self.validationHistory = []
            self.globalTrainingStep = 0
        last_grabvals = []
        for i in range(epochs):
            error = 0
            step = self.globalTrainingStep + i
            gvars = [self.error] + self.grabVars
            mbs = self.miniBatchSize
            ncases = len(cases)
            nmb = math.ceil(ncases / mbs)
            for cstart in range(0, ncases, mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases, cstart + mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                                 feed_dict=feeder, step=step, showInterval=self.showInterval)
                if self.showInterval and (step % self.showInterval == 0):
                    step = self.globalTrainingStep + i + 1

                error += grabvals[0]
                #last_grabvals = grabvals
            step = self.globalTrainingStep + i
            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step, sess)
        self.globalTrainingStep += epochs
        TFT.plot_training_history(self.error_history, self.validationHistory, xtitle="Epoch", ytitle="Error",
                                  title="History", fig=not(continued))


    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard loss function is used for testing.

    def do_testing(self, sess, cases, epoch = 'test', msg='Testing', bestk=1):
        inputs = [c[0] for c in cases]; targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        #test_func = self.error
        #if bestk is not None:
        testFunc = self.gen_match_counter(self.output, [TFT.one_hot_to_int(list(v)) for v in targets], k = 1)
        accuracy, _, _ = self.run_one_step(grabbed_vars = self.grabVars, probed_vars = self.probes,
                                           operators = testFunc, session = sess,
                                           feed_dict = feeder, showInterval = 0)
        loss, grabvals, _ = self.run_one_step(grabbed_vars=self.grabVars, probed_vars=self.probes,
                                              operators=self.error, session=sess,
                                              feed_dict=feeder, showInterval=0)
        accuracy = 100 * (accuracy / len(cases))
        if bestk is None:
            print('Epoch: %s\t\t %s loss: N/A\t\t %s : %s %%' % (epoch, msg, msg, accuracy))
        else:
            print('Epoch: %s\t\t %s loss: %5.4f\t\t %s accuracy: %d %%' % (epoch, msg, loss, msg, accuracy))
        return loss, grabvals  # self.error uses MSE, so this is a per-case value when bestk=None


    # Mapping: post-training activity that is designed to clarify the link between input patterns and hidden and/or output patterns.
    # USe the grabbed variables
    def do_mapping(self, session, cases):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        testFunc = self.gen_match_counter(self.output, [TFT.one_hot_to_int(list(v)) for v in targets], k=1)

        _, vals, _ = self.run_one_step(grabbed_vars = [self.mapVars]+ [self.dendrogramVars], probed_vars = self.probes,
                                           operators = testFunc, session = session,
                                           feed_dict = feeder, showInterval = 0)
        mapped_vals = vals[0]
        dendrogram_vals = vals[1]
        return mapped_vals, dendrogram_vals



    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k = 1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.caseMan.get_training_cases(), epochs, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.caseMan.get_testing_cases()
        if len(cases) > 0:
            _, grabvals = self.do_testing(sess, cases, msg='Final testing', bestk=bestk)
            if len(grabvals) > 0:
                self.display_grabvars(grabvals, self.grabVars, step = self.globalTrainingStep)

    def consider_validation_testing(self, epoch, sess):
        if self.validationInterval and (epoch % self.validationInterval == 0):
            cases = self.caseMan.get_validation_cases()
            if len(cases) > 0:
                error, _ = self.do_testing(sess, cases = cases, epoch = epoch, msg='Validation')
                self.validationHistory.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, cases = self.caseMan.get_training_cases(), epoch = 'test', msg='Final training', bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars = None, probed_vars = None, dir = 'probeview',
                     session = None, feed_dict = None, step = 1, showInterval = 1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict = feed_dict)
            sess.probe_stream.add_summary(results[2], global_step = step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict = feed_dict)
        if showInterval and (step % showInterval == 0):
            pass
            #self.display_grabvars(results[1], grabbed_vars, step=step)
            #self.display_loss_and_error()
        return results[0], results[1], sess

    def display_loss_and_error(self):
        pass


    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix
                pass
            elif type(v) == np.ndarray and len(v.shape) == 1: # if v is a vector (i.e. a bias vector)
                v = np.array([v]) # convert to matrix

            if self.grabVarPlotType == 'hinton':
                TFT.hinton_plot(v, fig=self.grabvarFigures[fig_index], title='Hinton plot of ' + names[i] + ' at step ' + str(step))
            else: #( self.grabVarPlotType == 'matrix' )
                TFT.display_matrix(v, fig=self.grabvarFigures[fig_index], title='Matrix of ' + names[i] + ' at step ' + str(step))
            fig_index += 1

    def run(self, showInterval = 100, validationInterval = 100, displayWeights = [], displayBiases = [],
            plot_type = 'hinton', epochs=100, sess=None, continued=False, bestk=None):
        self.showInterval = showInterval                        # Frequency of showing grabbed variables
        self.validationInterval = validationInterval

        self.grabVarPlotType = plot_type      # 'hinton' or 'matrix'

        self.displayWeights = displayWeights  # List of the weight arrays(their hidden layer indices) to be visualized at the end of the run
        self.displayBiases = displayBiases  # List of the bias vectors(their hidden layer indices) to be visualized at the end of the run
        # Add all grabbed variables
        for weight in self.displayWeights:
            self.add_grabvar(weight, 'wgt')
        for bias in self.displayBiases:
            self.add_grabvar(bias, 'bias')

        #PLT.ion()
        self.training_session(epochs, sess = sess, continued = continued)
        self.test_on_trains(sess = self.current_session, bestk = bestk)
        self.testing_session(sess = self.current_session, bestk = bestk)
        self.close_current_session(view = False)
        #PLT.ioff()

    def run_mapping(self, case_generator = None, mapBatchSize = 0, mapLayers = [], mapDendrograms = []):
        self.mapBatchSize = mapBatchSize        # Size of batch of cases used for a map test. 0 indicates no map test
        self.mapLayers = mapLayers              # List of layers(their indices) to be visualized during a map test
        self.mapDendrograms = mapDendrograms    # List of layers(their indices) whose activation patterns will be used to make dendrograms

        self.reopen_current_session()
        if self.mapBatchSize:
            # either a chosen set of cases or a random subset of the training cases:
            cases = case_generator() if case_generator else self.caseMan.get_mapping_cases(self.mapBatchSize)

            # Add the monitored variables
            self.add_mapvars()
            self.add_dendrogramvars()
            mapvals, dendrovals = self.do_mapping(session = self.current_session, cases = cases)

            # Plotting
            names = [x.name for x in self.mapVars]
            for i, v in enumerate(mapvals):
                if type(v) == np.ndarray and len(v.shape) > 1:  # If v is a matrix, use hinton plotting
                    TFT.hinton_plot(v, fig=None, title='Activation pattern of layer ' + names[i])
            if len(self.mapDendrograms) > 0:
                names = [x.name for x in self.dendrogramVars]
                labels = [TFT.bits_to_str(s[0]) for s in cases]
                #labels = [TFT.one_hot_to_int(c[1]) for c in cases]
                for (i, v) in enumerate(dendrovals):
                    TFT.dendrogram(v, labels, title = 'Dendrogram of ' + names[i])

        while (not PLT.waitforbuttonpress()):
            pass
        PLT.close('all')

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess = self.current_session, continued = True, bestk = bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self, ann, index, invariable, insize, outsize, activationFunc = 'relu', initWeightRange= (-1,1)):
        self.ann = ann
        self.insize = insize        # Number of neurons feeding into this module
        self.outsize = outsize      # Number of neurons in this module
        self.input = invariable     # Either the gann's input variable or the upstream module's output
        self.index = index
        self.activationFunc = activationFunc
        self.initialWeightRange = initWeightRange
        self.name = "Module-" + str(self.index)
        self.build()

    def build(self):
        moduleName = self.name
        n = self.outsize
        if self.ann.weightInitType == 'normalized':
            self.weights = tf.Variable(np.random.randn(self.insize,n)*math.sqrt(2.0/self.insize), name=moduleName+'-wgt', trainable=True)

        elif self.ann.weightInitType == 'uniform':
            (lower_w, upper_w) = self.initialWeightRange
            self.weights = tf.Variable(np.random.uniform(lower_w, upper_w, size=(self.insize,n)),
                                   name=moduleName+'-wgt', trainable=True) # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-0.01, 0.01, size=n),
                                  name=moduleName+'-bias', trainable=True)  # First bias vector

        # Set activation function for the neurons in the module
        if(self.activationFunc == 'relu'):
            self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=moduleName+'-out')
        elif (self.activationFunc == 'elu'):
            self.output = tf.nn.elu(tf.matmul(self.input, self.weights) + self.biases, name=moduleName + '-out')
        elif(self.activationFunc == 'sigmoid'):
            self.output = tf.nn.sigmoid(tf.matmul(self.input, self.weights) + self.biases, name=moduleName + '-out')
        elif(self.activationFunc == 'tanh'):
            self.output = tf.nn.tanh(tf.matmul(self.input, self.weights) + self.biases, name=moduleName + '-out')
        elif(self.activationFunc == 'softmax'):
            self.ann.rawOutput = tf.add(tf.matmul(self.input, self.weights), self.biases, name = moduleName + '-raw-out')
            self.output = tf.nn.softmax(tf.matmul(self.input, self.weights) + self.biases, name=moduleName + '-out')
        else:
            raise AssertionError('Unknown activation function ' + self.activationFunc + '.')

        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self, cfunc, cfrac = 1.0, vfrac = 0.0, tfrac = 0.0):
        self.casefunc = cfunc                                   # Function used to generate all data cases from a dataset
        self.case_fraction = cfrac                              # What fraction of the total data cases to use
        self.validation_fraction = vfrac                        # What fraction of the data to use for validation
        self.test_fraction = tfrac                              # What fraction of the data to use for final testing
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()


    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)       # create numpy array of the cases
        np.random.shuffle(ca)           # Randomly shuffle all cases
        if self.case_fraction < 1:
            case_separator = round(len(self.cases) * self.case_fraction)
            ca = ca[0:case_separator]   # only use a fraction of the cases

        training_separator = round(len(ca) * self.training_fraction)
        validation_separator = training_separator + round(len(ca) * self.validation_fraction)
        self.training_cases = ca[0:training_separator]
        self.validation_cases = ca[training_separator:validation_separator]
        self.testing_cases = ca[validation_separator:]

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases
    def get_mapping_cases(self, mapBatchSize):
        training_cases = np.array(self.training_cases)
        np.random.shuffle(training_cases)
        mapping_cases = training_cases[:mapBatchSize]
        return mapping_cases



def autoex(epochs=300, nbits=4, lrate=0.03, showint=100, mbs=None, vfrac=0.1, tfrac=0.1, vint=100,  bestk=1):
    size = 2**nbits
    mbs = mbs if mbs else size
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    cman = Caseman(cfunc = case_generator, vfrac = vfrac, tfrac = tfrac)
    ann = Gann(netDims = [size, nbits, size], cMan = cman, learningRate = lrate, mbs = mbs,
               hiddenActivationFunc = 'relu', outputActivationFunc = 'none')
    #ann.gen_probe(0, 'wgt', ('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    #ann.gen_probe(1, 'out', ('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0, 'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs = epochs, showInterval = showint, validationInterval = vint, bestk = bestk)
    ann.runmore(epochs*2, bestk = bestk)
    return ann


def countex(epochs=100, nbits=4, ncases=500, lrate=0.1, showint=500, mbs=10, cfrac = 1.0, vfrac=0.1, tfrac=0.1, vint=20, bestk=1):
    mapBatchSize = 2**nbits
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases,nbits))
    cman = Caseman(cfunc=case_generator, cfrac=cfrac, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(name = 'countex', netDims=[nbits, nbits*3, nbits*3, nbits+1], cMan=cman, learningRate=lrate, mbs=mbs,
               hiddenActivationFunc = 'relu', outputActivationFunc = 'softmax', lossFunc = 'softmax_cross_entropy',
               optimizer = 'momentum', momentum = 0.1, weightRange=(-.1,.1))

    ann.run(epochs = epochs, showInterval = showint, validationInterval = vint, displayBiases=[], displayWeights=[], plot_type = 'hinton', bestk = bestk)
    #TFT.plot_training_history(ann.error_history, ann.validationHistory, xtitle="Epoch", ytitle="Error",
                           #   title="training history", fig=True)

    # generate all possible input cases
    case_generator = (lambda: TFT.gen_vector_count_cases(mapBatchSize, nbits, random=False))
    ann.run_mapping(case_generator, mapBatchSize = mapBatchSize, mapLayers = [0], mapDendrograms = [2])


    return ann


#countex()