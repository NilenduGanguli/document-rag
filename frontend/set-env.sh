#!/bin/sh
set -e

echo "Exporting environment variables for Frontend..."

export API_URL="http://api:8000/api/v1"
export POSTGRES_SERVER="db"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_DB="kyc_rag"
export POSTGRES_PORT="5432"

# Execute the main container command (CMD)
exec "$@"