import numpy as np
import math 

class LocalLinearRegression():

    op_params = np.zeros((1,1))
    loss_holder = []

    def hypothesis(self, x, params):
        hypothesis = np.dot(x,params)
        return hypothesis

    def cost(self, x, y, params, mode):
        sample_size = len(x)
        
        if mode == "RMSE":
            cost_arr = np.square((self.hypothesis(x,params) - y))
            cost = np.sum(cost_arr)
            cost = (1/sample_size)*cost
            return math.sqrt(cost)

        elif mode == "MAE":
            cost_arr = abs(self.hypothesis(x,params) - y)
            cost = np.sum(cost_arr)
            cost = (1/(sample_size))*cost
            return cost

    def gradient(self, x, y, params, mode):
        sample_size = len(x)
        if mode == "RMSE":
            gradient_val = np.dot(x.T,self.hypothesis(x,params) - y)
            gradient_val = (1/(sample_size**1.0))*gradient_val
            gradient_val = (1/self.cost(x,y,params,"RMSE"))*gradient_val
            return gradient_val
        
        elif mode == "MAE":
            hyp = self.hypothesis(x,params) - y
            hyp = abs(hyp)/hyp
            gradient_val = np.dot(x.T,hyp)
            gradient_val = (1/sample_size)*(gradient_val)
            return gradient_val


    def gradientDescentRMSE(self, x, y, epochs = 1000, alpha = 0.1):
        sample_size = len(x)
        param_size = len(x[0])
        params = np.zeros((param_size,1))

        for i in range(0,epochs):
            self.loss_holder.append([params.copy(),self.cost(x,y,params,"RMSE")])
            for j in range(0,param_size):
                grad = self.gradient(x,y,params,"RMSE")
                params[j] = params[j] - alpha*grad[j]
        return params

    def gradientDescentMAE(self, x, y, epochs = 1000, alpha = 0.1):
        sample_size = len(x)
        param_size = len(x[0])
        params = np.zeros((param_size,1))

        for i in range(0,epochs):
            self.loss_holder.append([params.copy(),self.cost(x,y,params,"MAE")])
            for j in range(0,param_size):
                grad = self.gradient(x,y,params,"MAE")
                params[j] = params[j] - alpha*grad[j]
            
        return params

    def normalEquation(self, x, y):
        a = np.dot(x.T,x)
        a = np.linalg.inv(a)
        b = np.dot(x.T,y)
        return np.dot(a,b)

    def fit(self, x, y, mode = "RMSE", epochs = 1000, alpha = 0.1):
        x = np.insert(x, 0, 1, axis=1)
        y = y[ : , np.newaxis]

        if mode == "RMSE":
            params = self.gradientDescentRMSE(x,y,epochs,alpha)
            self.op_params = params

        elif mode == "MAE":
            params = self.gradientDescentMAE(x,y,epochs,alpha)
            self.op_params = params
        return self

    def predict(self, x):
        x = np.insert(x, 0, 1, axis = 1)
        params = self.op_params.flatten()
        params = params[:, np.newaxis]
        y_pred = self.hypothesis(x, params).flatten()
        return y_pred