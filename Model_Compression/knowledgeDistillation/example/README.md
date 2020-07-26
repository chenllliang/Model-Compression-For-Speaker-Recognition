# An example of knowledge distillation

## Configuration
* Teacher Model : CNN parameters num: 1007590 
* Student Model : FCN parameters num: 709110
see model.py
* Dataset : MNIST

* Distillation Temperature: 100
* Distillation Alpha: 0.2

## Steps
0. requirements: torch==1.4.0, torchvision==0.5.0
1. First edit train.py set the right path of MNIST or comment the class MNIST(if you can connect to the source) and run
 ```python train.py ``` first.

 if everything goes right, you will get 
 ```
 Start training student net alone
[1,  2000] loss: 1.029
[1,  4000] loss: 0.367
[1,  6000] loss: 0.300
[2,  2000] loss: 0.250
[2,  4000] loss: 0.218
[2,  6000] loss: 0.199
Student Finished Training
9484
Accuracy of the student network without teacher on the 10000 test images: 0.9484
Start training teacher net 
[1,  2000] loss: 0.653
[1,  4000] loss: 0.199
[1,  6000] loss: 0.145
[2,  2000] loss: 0.104
[2,  4000] loss: 0.093
[2,  6000] loss: 0.081
Teacher Finished Training
9793
Accuracy of the teacher networkon the 10000 test images: 0.9793
 ```

 and the weights of teacher net `teacher.pth`

 2. run `python distillation.py`, you will get 
 ```
 [1,  2000] loss: 1.244
[1,  4000] loss: 0.459
[1,  6000] loss: 0.348
[2,  2000] loss: 0.281
[2,  4000] loss: 0.229
[2,  6000] loss: 0.191
Student Finished Training
9525
Accuracy of the student network with teacher on the 10000 test images: 0.9525
 ```

## Result(acc)

| Model | Acc  
| :-----:| :-----:
| Teacher Net | 0.9793 
| Student Net(alone) | 0.9484 
| Student Net(distilled) | 0.9525 



