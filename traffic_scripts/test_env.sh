# For the bottleneck-misweighting experiment, run
python traffic_scripts/traffic_env.py \
    singleagent_bottleneck \
    --name="testing" \
    --reward_fn="desired_vel,forward,lane_bool" \
    --reward_weighting="1,0.1,0.01" \
    --test
