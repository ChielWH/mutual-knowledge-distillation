This is my thesis project, I combine Deep mutual learning and teaching assistant knowledge distillation

#### Deep mutual learning:
Description: DML is used to train a set of models, where each model incorporates a learning signal of the predictions from the other models in its loss **during training**. <br>
[Original paper](https://arxiv.org/pdf/1706.00384.pdf)<br>
[Author's code](https://github.com/YingZhangDUT/Deep-Mutual-Learning)

#### Teaching assistant knowledge distillation:
Description: Knowledge distillation first trains a high capacity, well performing model (teacher) of which, after fully training it, its predictions are used as a learning signal to incorporate to the loss of a lower capacity model (student). Teaching assistants refer to inbetween, medium capacity models, the first serve as a student after which it serves as a teacher to the small capacity model.<br>
[Original paper](https://arxiv.org/pdf/1902.03393.pdf)<br>
[Author's code](https://github.com/imirzadeh/Teacher-Assistant-Knowledge-Distillation)

This repo is based on the PyTorch implementation of Deep mutual learning from [this repo](https://github.com/chxy95/Deep-Mutual-Learning)
