#!/bin/sh
set -e

# In a production environment, this script could be extended to fetch secrets 
# dynamically from Azure Key Vault, AWS Secrets Manager, or HashiCorp Vault 
# before starting the application.

echo "Exporting environment variables for OCR Service..."

# OCR Service Configuration
export PORT="8001"
export HOST="0.0.0.0"

# Execute the main container command (CMD)
exec "$@"
