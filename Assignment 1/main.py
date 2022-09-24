# Write your assignment here
import numpy as np 
import argparse
import json

if __name__ == '__main__':

    class Linear_Regression:
        def __init__(self, infilepath, jsonfilepath):
            #load dataset from relative file path
            self.dataset = np.loadtxt(infilepath) 
            
            #5, the number of training samples in the dataset
            self.numSamples = self.dataset.shape[0] 
            
            #Create a column array for (x1, x2) - 1st and 2nd column
            #Create a column array for y - last column
            self.xs = self.dataset[:, 0:self.dataset.shape[1]-1] 
            self.ys = self.dataset[:, self.dataset.shape[1]-1] 
            
            #Create a column array of "1s" with dimension of numSamples - first column of the "BigPhi" matrix
            phi0 = np.ones((self.dataset.shape[0],1)) 
            
            #Append "phi0" and "xs" to form the "BigPhi" matrix)
            self.BigPhi = np.append(phi0, self.xs, axis=1) 
            
            #W_analytical and w_GD properties
            self.w_GD = np.ones(self.xs.shape[1]+1)
            self.w_analytical = np.zeros(self.xs.shape[1]+1)
            
            #Load hyperparameters from .json file
            jsonfile = open(jsonfilepath) 
            param = json.load(jsonfile)
            self.LearningRate = param['learning rate']
            self.numIterations = param['num iter']
            
        #Calculate wstar = w_analytical accroding to lecture 4 notes
        def wstar(self):
            self.w_analytical = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.BigPhi.T,self.BigPhi)),self.BigPhi.T),self.ys)
        
        #Helps with summing gradient over all samples
        def BGD_Dot(self, j,i):
            return (np.dot(self.w_GD,self.BigPhi[j])-self.ys[j])*self.BigPhi[j][i]

        #BGD method calculations that iterates and updates
        def BGD_Loop(self):
            cache = np.zeros(self.w_GD.size)
            for i in range(self.w_GD.size):
                gradSum = 0
                for j in range(self.numSamples):
                    gradSum += self.BGD_Dot(j,i)
                cache[i] = -(self.LearningRate/self.numSamples) * gradSum
            self.w_GD += cache
           
        def BGD(self):
            for i in range(self.numIterations):
                self.BGD_Loop()
    
    #Parser to receive .in file and .json file
    parser = argparse.ArgumentParser(description='Receive 2 inputs or 2 relative file path for the .in file and .json file')
    parser.add_argument('infilepath',  help='.in filepath')
    parser.add_argument('jsonfilepath', help = '.json filepath')
    args = parser.parse_args()

    #Training using simple linear regression model for both analytical wstar + batch gradient descent method
    Lin_Reg = Linear_Regression(args.infilepath, args.jsonfilepath)
    Lin_Reg.wstar()
    Lin_Reg.BGD()
    
    #Obtain file paths from arguments and replace .in with .out for the extentions
    infilepath = args.infilepath
    outfilepath = infilepath.replace('.in','.out')
    outfile = open(outfilepath, 'wt')
    
    #Output W_analytical and W_GD results as strings
    outputText = "w_analytical\n"
    for i in range(Lin_Reg.w_GD.size):
        outputText += (str(Lin_Reg.w_analytical[i]) + "\n")
    
    outputText += "w_GD\n"
    for i in range(Lin_Reg.w_GD.size):
        outputText += (str(Lin_Reg.w_GD[i]) + "\n")

    outfile.write(outputText)
    outfile.close()