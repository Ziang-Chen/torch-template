#-----------------------------------------------------------
#   Data Loader Base Class
#----------------------------------------------------------
# 



import pickle


class DataLoader:
    def __init__(self,loader_func=None,path=None):
        self.data=None
        if not (path==None):
            if loader_func==None:
                with open(path,'rb') as f:
                    self.data=pickle.load(f)
            else:
                self.data=loader_func(path)
        #return self.data

    def __getitem__(self,item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class Stream:
    def __init__(self,x:DataLoader,y:DataLoader):
        self.x=x
        self.y=y
        assert len(self.x)==len(self.y),"X|Y Size not match"

    def vanila_batch(self,batch_size):
        num_of_iteration=int(len(self.x)/batch_size)
        for i in range(num_of_iteration):
            yield self.x[i*batch_size:(i+1)*batch_size],self.y[i*batch_size:(i+1)*batch_size]
