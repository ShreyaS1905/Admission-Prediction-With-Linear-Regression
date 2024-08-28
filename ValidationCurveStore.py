import numpy as np
import matplotlib.pyplot as plt
from Kfold import KFold
from LocalLinearReg import LocalLinearRegression


class ValidationCurve:
    
    def getLoss(self, x_train, x_val, y_train, y_val, mode, alpha, epochs, index):
        linear_reg = LocalLinearRegression()
        linear_reg.loss_holder = []
        linear_reg.fit(x_train, y_train, mode, epochs, alpha)
        
        x_val = np.insert(x_val, 0, 1, axis=1)
        y_val = y_val[ : , np.newaxis]

        train_loss = np.zeros(len(linear_reg.loss_holder))
        val_loss = np.zeros(len(linear_reg.loss_holder))

        for i in range (0,len(linear_reg.loss_holder)):
            store = linear_reg.loss_holder[i]
            params = store[0]
            train_loss[i] = store[1]
            val_loss[i] = linear_reg.cost(x_val, y_val, params, mode)

        self.train_loss.append(train_loss[-1])
        self.val_loss.append(val_loss[-1])
        self.params.append(linear_reg.op_params)

        print("\nFor fold => "+str(index))
        print("=> Training loss => "+str(train_loss[-1]))
        print("=> Validation loss => " +str(val_loss[-1]))

        return train_loss, val_loss

    def plot_curve(self, n_splits, x, y, mode, alpha, epochs):
        kf = KFold(n_splits)
        train_op = 0
        val_op = 0
        ind = 1
        self.train_loss = []
        self.val_loss = []
        self.params = []

        for train_idx, val_idx in kf.split(x):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            train_loss, val_loss = self.getLoss(x_train, x_val, y_train, y_val, mode, alpha, epochs,ind)
            train_op = train_op + train_loss
            val_op = val_op + val_loss
            ind = ind + 1
        
        train_op = train_op/n_splits
        val_op = val_op/n_splits

        epoch_range = [i for i in range(0,epochs)]
        plt.figure(figsize=(5,5))
        plt.plot(epoch_range,val_op,'r',label = 'Validation Loss')
        plt.plot(epoch_range,train_op,'b',label = 'Training Loss')

        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Train Loss and Validation Loss v/s Epochs (mode = "+mode+" alpha = "+str(alpha)+")")
        plt.legend()
        plt.show()