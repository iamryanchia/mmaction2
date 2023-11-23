# python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
#     --seed=0 # --deterministic

cd mmaction2

(
    cd ../aigc-data
    unzip sports.zip
)

bash tools/dist_train.sh configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_sports.py 4

# python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_sports.py \
#     --seed=0 # --deterministic
