#!/bin/bash
# Quick start script for 3D Measurement System Docker deployment
# Usage: ./docker-run.sh [start|stop|restart|logs|status|build]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.gpu.yml"
CONTAINER_NAME="3d-measurement-gpu"
IMAGE_NAME="3d-measurement:gpu-v2.0"

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed!"
        echo "Please install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_success "Docker is installed"
}

check_docker_compose() {
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    print_success "Docker Compose is installed"
}

check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime not available!"
        echo "Please install NVIDIA Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    print_success "NVIDIA Docker runtime is available"
}

create_directories() {
    mkdir -p output logs data
    chmod -R 755 output logs data
    print_success "Created necessary directories"
}

start_service() {
    print_info "Starting 3D Measurement System..."
    
    check_docker
    check_docker_compose
    check_nvidia_docker
    create_directories
    
    docker compose -f $COMPOSE_FILE up -d
    
    print_success "Container started successfully!"
    print_info "Waiting for service to be ready (this may take 60 seconds)..."
    
    # Wait for health check
    for i in {1..60}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "Service is ready!"
            echo ""
            echo "API available at: http://localhost:8000"
            echo "API docs: http://localhost:8000/docs"
            echo ""
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    echo ""
    print_error "Service failed to become ready. Check logs with: ./docker-run.sh logs"
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
        echo "=== GPU Status ==="
        docker exec $CONTAINER_NAME nvidia-smi 2>/dev/null || print_error "Could not check GPU status"
        
        echo ""
        echo "=== Resource Usage ==="
        docker stats --no-stream $CONTAINER_NAME
    else
        print_error "Container is not running"
    fi
}

build_image() {
    print_info "Building Docker image..."
    check_docker
    docker compose -f $COMPOSE_FILE build --no-cache
    print_success "Image built successfully!"
}

test_api() {
    print_info "Testing API endpoints..."
    
    # Test health
    echo -n "Testing /health... "
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "OK"
    else
        print_error "FAILED"
        return 1
    fi
    
    # Test GPU stats
    echo -n "Testing /gpu-stats... "
    if curl -f http://localhost:8000/gpu-stats &> /dev/null; then
        print_success "OK"
    else
        print_error "FAILED"
        return 1
    fi
    
    print_success "All tests passed!"
}

shell_access() {
    print_info "Opening shell in container..."
    docker exec -it $CONTAINER_NAME bash
}

show_help() {
    cat << EOF
3D Measurement System - Docker Management Script

Usage: ./docker-run.sh [COMMAND]

Commands:
    start       Start the measurement system
    stop        Stop the measurement system
    restart     Restart the measurement system
    logs        Show and follow logs
    status      Show container and service status
    build       Build Docker image
    test        Test API endpoints
    shell       Open bash shell in container
    help        Show this help message

Examples:
    ./docker-run.sh start       # Start the service
    ./docker-run.sh logs        # View logs
    ./docker-run.sh status      # Check status
    ./docker-run.sh test        # Test API

For more information, see DOCKER_GUIDE.md
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
    build)
        build_image
        ;;
    test)
        test_api
        ;;
    shell)
        shell_access
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

