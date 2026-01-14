
conda create --name edu-cognition python=3.9

pip install dlib
pip install mediapipe

conda activate edu-cognition



#first experiment
# Inception frame-level feature  +  temporal pooling (transformer encoder)+ classifier
# use model_inception_v1.py , no position encoding
python train.py --model_type v1
python test.py --model_type v1

Task B Accuracy: 0.4615
Task E Accuracy: 0.5216
Task C Accuracy: 0.7014
Task F Accuracy: 0.7852


# v2= v1+ Position Encoding 
python train.py --model_type v2
python test.py --model_type v2

Task B Accuracy: 0.4627
Task E Accuracy: 0.5230
Task C Accuracy: 0.7031
Task F Accuracy: 0.7846


# v3= v2 + MoE 
python train.py --model_type v3
python test.py --model_type v3

Task B Accuracy: 0.3684
Task E Accuracy: 0.5255
Task C Accuracy: 0.7016
Task F Accuracy: 0.7861

#TODO1  explore different MoE dedsign to improve
#TODO2  combine AUs and VA to train v2