# Instructions for Running Homework

All deliverables turned in for the homework can be generated using the instruction provided by the assignment description.

A couple of things to not about changes I made to the argument are as follow:
1. In training_model.py, instead of --max-iter i am using --max_epoch which dictates the number of epochs for the model to train for
2. In training_model.py, I have included --with_eval which, when called, evaluates the model at the end of each epoch
3. In eval_model.py, I have included --vis which, when called, will only save outputs instead of evaluating on the entire evaluation loader
4. In eval_model.py, I utilize a global variable "ids" which is used to define 3 ids to save the output from while evaluating

The calls I used for section 1 are exactly as provided in the assignment description.

The specific calls I used to train my three models for section 2 are as follows:
1. python3 train_model.py --type 'vox' --with_eval --max_epoch 75
2. python3 train_model.py --type 'point' --with_eval --max_epoch 40
3. python3 train_model.py --type 'mesh' --with_eval --w_smooth 5.0 --max_epoch 25