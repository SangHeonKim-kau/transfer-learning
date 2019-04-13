# transfer-learning
Imagenet resnet50 use to facial expression recognition
(利用ImageNet预训练的模型resnet50去给人脸7种表情做分类)

Download fer2013 and transter(see my github for trans) it to img 

place it in emotion_data files

数据要做清洗：face detect->face alignment->classification

此训练用到了tensorboardX将训练可视化

由于resnet18计算量较大，接下来换网络模型为mobilenet 或者shufflenetv2，再将bn层和卷积层融合，进一步提高推断时的效率
