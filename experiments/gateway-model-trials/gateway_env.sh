# Source this before running the scripts here: points the recon gateway at the local LiteLLM proxy (../../gateway/).
set -a
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../../gateway/litellm/.env.local"
RECON_GATEWAY_KEY=$LITELLM_MASTER_KEY
RECON_LLM_URL=http://127.0.0.1:4000/v1
RECON_CLASSIFIER_URL=http://127.0.0.1:4000/recon/classifier
RECON_KAI_URL=http://127.0.0.1:4000/recon/kai
RECON_CLEF_URL=http://127.0.0.1:4000/recon/clef
RECON_CLEF_FLASH_URL=http://127.0.0.1:4000/recon/clef-flash
RECON_D3_URL=http://127.0.0.1:4000/recon/d3
RECON_D3_RTX_URL=http://127.0.0.1:4000/recon/d3-rtx
RECON_D3_MINI_URL=http://127.0.0.1:4000/recon/d3-mini
RECON_D3_FLASH_URL=http://127.0.0.1:4000/recon/d3-flash
set +a
