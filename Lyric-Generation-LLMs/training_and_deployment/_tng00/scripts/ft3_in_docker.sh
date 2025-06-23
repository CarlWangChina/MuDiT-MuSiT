SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
source "$PARENT_DIR/.env.config"

TRNG_DATA_FN=rdy_cb240327063523-0.json
CF_TRNG_DATA=$CP_TRNG_DATA/$TRNG_DATA_FN
HP_CKPT_OP_D=$HP_FT_OUTPUT/ckpt3201_cb

GPU_DEVICE='"device=0,1,2,3,4,5"'

mkdir -p $HP_CKPT_OP_D

docker run --gpus ${GPU_DEVICE} --rm --name qwen-ft3 \
    --mount type=bind,source=${H_BASE_MODEL},target=${C_BASE_MODEL} \
    --mount type=bind,source=${HP_TRNG_DATA},target=${CP_TRNG_DATA} \
    --mount type=bind,source=${HP_CKPT_OP_D},target=${CP_FT_OUTPUT} \
    -v $HP_CUST_SCRS:$CP_CUST_SCRS \
    --shm-size=2gb -itd ${DEF_DOCKER_IMG} \
    bash cs_scripts/ft3_qlora_ds.sh -m ${C_BASE_MODEL}/ -d ${CF_TRNG_DATA}