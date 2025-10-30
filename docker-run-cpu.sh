#!/bin/bash
# Quick start script for CPU-only version (no GPU required)
# Usage: ./docker-run-cpu.sh [start|stop|restart|logs|status]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
COMPOSE_FILE="docker-compose.yml"
CONTAINER_NAME="3d-measurement-cpu"
IMAGE_NAME="3d-measurement:cpu-v2.0"

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        exit 1
    fi
    print_success "Docker is installed"
}

check_docker_compose() {
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    print_success "Docker Compose is installed"
}

create_directories() {
    mkdir -p output logs data
    chmod -R 755 output logs data
    print_success "Created necessary directories"
}

start_service() {
    print_warning "⚠ WARNING: Starting CPU-only version. Performance will be VERY slow!"
    print_warning "⚠ For production, install NVIDIA Docker and use ./docker-run.sh"
    echo ""
    print_info "Starting 3D Measurement System (CPU-only)..."
    
    check_docker
    check_docker_compose
    create_directories
    
    docker compose -f $COMPOSE_FILE up -d
    
    print_success "Container started!"
    print_info "Waiting for service to be ready (this may take 60 seconds)..."
    
    # Wait for health check
    for i in {1..60}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "Service is ready!"
            echo ""
            echo "API available at: http://localhost:8000"
            echo "API docs: http://localhost:8000/docs"
            echo ""
            print_warning "Note: CPU-only mode will be 10-50x slower than GPU version"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    echo ""
    print_error "Service failed to become ready. Check logs with: ./docker-run-cpu.sh logs"
    exit 1
}

stop_service() {
    print_info "Stopping 3D Measurement System..."
    docker compose -f $COMPOSE_FILE down
    print_success "Container stopped"
}

restart_service() {
    stop_service
    sleep 2
    start_service
}

show_logs() {
    docker compose -f $COMPOSE_FILE logs -f
}

show_status() {
    echo "=== Container Status ==="
    docker compose -f $COMPOSE_FILE ps
    echo ""
    
    if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" | grep -q $CONTAINER_NAME; then
        print_success "Container is running"
        
        echo ""
        echo "=== Health Check ==="
        if curl -f http://localhost:8000/health 2>/dev/null; then
            print_success "API is responding"
        else
            print_error "API is not responding"
        fi
        
        echo ""
        echo "=== Resource Usage ==="
        docker stats --no-stream $CONTAINER_NAME
    else
        print_error "Container is not running"
    fi
}

show_help() {
    cat << EOF
3D Measurement System - CPU-Only Version

⚠ WARNING: This is the CPU-only version (NO GPU required)
Performance will be 10-50x slower than GPU version.
For production use, install NVIDIA Docker and use ./docker-run.sh

Usage: ./docker-run-cpu.sh [COMMAND]

Commands:
    start       Start the measurement system (CPU-only)
    stop        Stop the measurement system
    restart     Restart the measurement system
    logs        Show and follow logs
    status      Show container and service status
    help        Show this help message

Examples:
    ./docker-run-cpu.sh start       # Start CPU-only version
    ./docker-run-cpu.sh logs        # View logs
    ./docker-run-cpu.sh status      # Check status

To use GPU version:
    1. Install NVIDIA Docker: ./install-nvidia-docker.sh
    2. Use GPU script: ./docker-run.sh start
EOF
}

# Main script
case "${1:-help}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

