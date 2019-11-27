import numpy as np 
import matplotlib.pyplot as plt
import sys
import pandas as pd
np.random.seed(1220)

class SupportVectorMachine_Peagsos:
    def __init__(self,trainx,trainy,lamdaa,batch_size=None):
        '''
         shape of trainx - n_samples*n_features
         shape of trainy - n_samples*1
         shape of weight - n_features*(n_classes*(n_classes-1))
         shape of bias  - 1*(n_classes*(n_classes-1))
        '''
        self.trainx = self.preprocess(trainx)
        self.weight = None
        self.bias= None
        self.trainy=trainy
        self.lamdaa=lamdaa
        # self.loss_hist=[]
        self.classes_dict={}
        if(batch_size is None):
            self.batch_size =1
        else:
            self.batch_size = batch_size
    
    def loss_fxn(self,x,y,w,b,batch_size=None):
        '''
        shape of x - n_samples*n_features
        shape of w - n_features*1
        shape of b - n_samples*1
        shape of y - n_samples*1
        '''
        term1 = (self.lamdaa/2)*np.matmul(w.T,w)
        epsillon = 1-y*(np.dot(x,w)+b)
        if(batch_size is None):
            term2 = (sum(epsillon[epsillon>0]))
        else:
            term2 = (sum(epsillon[epsillon>0])/batch_size)
        return term1 +term2
    
    def fit(self,num_iterations):
        classes = sorted(list(set(self.trainy)))
        self.weight = np.zeros((self.trainx.shape[1],int(len(classes)*(len(classes)-1)/2)))
        self.bias = np.ones((1,int(len(classes)*(len(classes)-1)/2)))
        for i in range(self.weight.shape[1]):
            self.weight[:,i].fill(np.sqrt(1/self.lamdaa/self.trainx.shape[1]))
            random_ind = np.random.randint(0,self.trainx.shape[1],size=int(self.trainx.shape[1]/2))
            self.weight[random_ind,i] = -self.weight[random_ind,i]

        wt_ind=0
        for cls in range(len(classes)-1):
            for cls2 in range(cls+1,len(classes)):
                # temp_loss_hist =[]
                curr_w = self.weight[:,wt_ind]
                curr_bias = self.bias[:,wt_ind]
                trainx_sub = self.trainx[np.logical_or(self.trainy==cls,self.trainy==cls2)]
                trainy_sub = self.trainy[np.logical_or(self.trainy==cls,self.trainy==cls2)]
                trainy_sub[trainy_sub==cls] = -1
                trainy_sub[trainy_sub==cls2] =1
                print(trainx_sub.shape,trainy_sub.shape)
                train_sub =np.hstack([trainx_sub,trainy_sub.reshape((trainy_sub.shape[0],1))])
                np.random.shuffle(train_sub)
                trainx_sub = train_sub[:,:-1]
                trainy_sub = train_sub[:,-1].flatten()
                print(trainx_sub.shape,trainy_sub.shape)
                print('ok')
                self.classes_dict[wt_ind] = {-1:cls,1:cls2}
                for i in range(1,num_iterations*trainx_sub.shape[0]):
                    constant = 1.0/(self.lamdaa*(i))
                    if(self.batch_size==1):
                        ind = np.random.randint(0,trainx_sub.shape[0],size=1)
                        x_temp,y_temp = trainx_sub[ind,:],trainy_sub[ind]
                        if(y_temp*(np.dot(x_temp,curr_w)+curr_bias)<1):
                            weight  = (1-constant*self.lamdaa)*curr_w + (constant)*np.dot(y_temp,x_temp)
                        else:
                            weight = (1-constant*self.lamdaa)*curr_w
                        bias = curr_bias + constant*(y_temp)
                    else:
                        x_temp,y_temp = self.get_batch(trainx_sub,trainy_sub)
                        inds = np.argwhere(y_temp*(np.dot(x_temp,curr_w)+curr_bias)<1).flatten()
                    
                        x_sub = x_temp[inds,:]
                        y_sub =y_temp[inds]

                        weight  = (1-constant*self.lamdaa)*curr_w + (constant/self.batch_size)*np.dot(y_sub,x_sub)
                        bias = curr_bias + (constant/self.batch_size)*sum((y_sub))
                        curr_bias = bias
                    mini = np.array([1,1/np.sqrt(self.lamdaa)/np.sqrt(np.sum(weight**2))])
                    new_w = np.amin(mini)*weight

                    # temp_loss_hist.append(self.loss_fxn(,self.trainy[self.trainy in [cls1,cls2]],))

                    if sum((new_w - curr_w)**2) <0.01:
                        print('optimized for',cls,cls2)
                        break
                    else:
                        curr_w = new_w
                self.weight[:,wt_ind] = curr_w
                self.bias[:,wt_ind] = curr_bias
                wt_ind+=1
        return self.weight,self.bias



    def get_batch(self,x,y):
        X_1 = x[np.argwhere(y==1).flatten(),:]
        X_2 = x[np.argwhere(y==-1).flatten(),:]
        Y_1 = y[np.argwhere(y==1).flatten()]
        Y_2 =y[np.argwhere(y==-1).flatten()]
        ind1 = np.random.randint(0,high=X_1.shape[0],size=self.batch_size)
        ind2 = np.random.randint(0,high=X_2.shape[0],size=self.batch_size)

        x_temp = np.concatenate((X_1[ind1,:],X_2[ind2,:]))
        y_temp = np.concatenate((Y_1[ind1],Y_2[ind2]))
        y_temp = y_temp.reshape((y_temp.shape[0],))
        return x_temp,y_temp
        
    
    def preprocess(self,x):
        new_X = x
        stds = new_X.std(axis=0)
        means = new_X.mean(axis=0)
        new_X = (new_X - means) / (stds+1e-100)

        return new_X

    def predict(self,x):
        '''
        @shape of x- n_samples*n_features
        '''
        prediction = np.matmul(x,self.weight)+self.bias
        output = np.zeros(prediction.shape,dtype=np.uint8)
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if(prediction[i][j]>0):
                    output[i][j] = int(self.classes_dict[j][1])
                else:
                    output[i][j] = int(self.classes_dict[j][-1])
        
        return output

    def majority_counter(self,array_2d):
        output_ = []
        for i in range(array_2d.shape[0]):
            vals,cnts = np.unique(array_2d[i,:],return_counts=True)
            output_.append(vals[cnts.argmax()])
        return output_





if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    pred_file = sys.argv[3]
    df = pd.read_csv(train_file,header=None).values
    df_test = pd.read_csv(test_file,header=None).values
    trainx = df[:,:784]
    trainy = df[:,784].reshape((trainx.shape[0],))

    testx= df_test[:,:784]
    SVM = SupportVectorMachine_Peagsos(trainx,trainy,1e-4,64)
    SVM.fit(1000)
    testx = SVM.preprocess(testx)
    pred = SVM.predict(testx)
    pred = SVM.majority_counter(pred)

    with open(pred_file,"w") as f:
        for o in pred:
            f.write(str(o))
            f.write("\n")


