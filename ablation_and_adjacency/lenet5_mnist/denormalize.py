#replace the return statement with code to denormalize each input feature and return results as a list.
#if you do not intend to denormalize, do not alter this file.

def denormalize(inputfeatures):
    temp = [x * 255 for x in inputfeatures]
    return temp