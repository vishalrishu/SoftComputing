import numpy as np

input1=np.array([1,1,1,1,1,1]).reshape(6,1) 
input2=np.array([-1,-1,-1,-1,-1,-1]).reshape(6,1)
input3=np.array([1,-1,-1,1,1,1]).reshape(6,1) 
input4=np.array([1,1,-1,-1,-1,-1]).reshape(6,1)

output1=np.array([1,1,1]).reshape(1,3)
output2=np.array([-1,-1,-1]).reshape(1,3)
output3=np.array([-1,1,1]).reshape(1,3)
output4=np.array([1,-1,1]).reshape(1,3)

set_A = np.concatenate((input1,input2,input3,input4),axis=1)
set_B = np.concatenate((output1,output2,output3,output4),axis=0)

weight = np.dot(set_A, set_B)

y=np.dot(input1.T,weight)
y[y<-1]=-1
y[y>1]=1

y=np.dot(input2.T,weight)
y[y<-1]=-1
y[y>1]=1

y=np.dot(output1,weight.T)
y[y<-1]=-1
y[y>1]=1

print(output2,"*",weight.T)
y=np.dot(output2,weight.T)

y[y<-1]=-1
y[y>1]=1
print("y=",y)
