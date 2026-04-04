#!/bin/bash

# =============================================================================
# permission.sh
# -----------------------------------------------------------------------------
# Purpose: Fix file ownership and permission issues in the current directory
# -----------------------------------------------------------------------------
# Description:
#   This script recursively changes the ownership of all files and directories
#   in the current working directory to the current user and their primary group.
#   It also grants the owner read, write, and execute permissions recursively.
# -----------------------------------------------------------------------------
# Usage:
#   bash permission.sh
# -----------------------------------------------------------------------------
# Parameters:
#   None
# -----------------------------------------------------------------------------
# Globals:
#   None
# -----------------------------------------------------------------------------
# Notes:
#   - Requires sudo privileges to change ownership
#   - Use with caution as it modifies permissions for all files recursively
# =============================================================================
sudo chown -R $(whoami):$(id -gn) . && chmod -R u+rwx .
