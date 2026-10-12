# Source this before running the scripts here: points the recon gateway at the local LiteLLM proxy (../../gateway/).
set -a
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../gateway/litellm/.env.local"
RECON_GATEWAY_KEY=$LITELLM_MASTER_KEY
RECON_LLM_URL=http://127.0.0.1:4000/v1
RECON_CLASSIFIER_URL=http://127.0.0.1:4000/recon/classifier
RECON_KAI_URL=http://127.0.0.1:4000/recon/kai
RECON_CLEF_URL=http://127.0.0.1:4000/recon/clef
RECON_CLEF_FLASH_URL=http://127.0.0.1:4000/recon/clef-flash
set +a
