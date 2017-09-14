rowValue = [1] * self.cols
                    rowIndex = 0
                    rowSpecIndex = 0
                    for i, element in enumerate(composition):
                        if(i == 0):
                            rowValue[rowIndex:rowIndex + element] = [0] * element
                        else:
                            rowValue[rowIndex:rowIndex + element + 1] = [0] * (element + 1)
                            rowIndex += 1
                        if(i < len(composition) - 1):
                            rowIndex += element + rowSpec[rowSpecIndex]
                            rowSpecIndex += 1