:construction: Under construction :construction:
## Mutual knowledge distillation
This is the code base of my thesis project called mutual knowledge distillation where I combine Deep mutual learning and teaching assistant knowledge distillation.

#### Deep mutual learning:
Deep mutual learning is used to train a set of models, where each model incorporates a learning signal of the predictions from the other models in its loss **during training**. <br>
[Original paper](https://arxiv.org/pdf/1706.00384.pdf), [author's code](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

#### Teaching assistant knowledge distillation:
Knowledge distillation first trains a high capacity, well performing model (teacher) of which, **after fully training** it, its predictions are used as a learning signal to incorporate to the loss of a lower capacity model (student). Teaching assistants (TA's) refer to inbetween, medium capacity models. TA's first serve as a student after which it becomes the teacher to a smaller capacity model.<br>
[Original paper](https://arxiv.org/pdf/1902.03393.pdf), [author's code](https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation)

This repo is based on the PyTorch implementation of Deep mutual learning from [this repo](https://github.com/chxy95/Deep-Mutual-Learning)
