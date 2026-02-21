#!/bin/sh
set -e

# In a production environment, this script could be extended to fetch secrets 
# dynamically from Azure Key Vault, AWS Secrets Manager, or HashiCorp Vault 
# before starting the application.

echo "Exporting environment variables..."

# Database Configuration
export POSTGRES_SERVER="db"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_DB="kyc_rag"
export POSTGRES_PORT="5432"

# Redis / Celery Configuration
export REDIS_HOST="redis"
export REDIS_PORT="6379"

# OCR Service
export OCR_SERVICE_URL="http://ocr_service:8001/ocr"

# Execute the main container command (CMD)
exec "$@"
