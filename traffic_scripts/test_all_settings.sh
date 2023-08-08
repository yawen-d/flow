# For the bottleneck-misweighting experiment, run
python traffic_scripts/traffic_env.py \
    singleagent_bottleneck \
    --name="testing" \
    --reward_fn="desired_vel,forward,lane_bool" \
    --reward_weighting="1,0.1,0.01"

# For the merge-misweighting experiment, run
python traffic_scripts/traffic_env.py \
    singleagent_merge \
    --name="testing" \
    --reward_fn="vel,accel,headway" \
    --reward_weighting="1,0.01,0.1"

# For the merge-ontological experiment, run
python traffic_scripts/traffic_env.py \
    singleagent_merge_bus \
    --name="testing" \
    --reward_fn="vel,accel,headway" \
    --reward_weighting="1,1,0.1"

# For the merge-scope experiment, run
python traffic_scripts/traffic_env.py \
    singleagent_merge \
    --name="testing" \
    --reward_fn="local_first,accel,headway" \
    --reward_weighting="1,1,0.1"
