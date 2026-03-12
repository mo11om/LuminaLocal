#!/bin/bash
#
# CV Matcher Kubernetes Deployment Helper
# Simplifies deployment, monitoring, and cleanup
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="default"
VLLM_DEPLOYMENT="vllm-deployment"
CV_MATCHER_JOB="cv-matcher-job"
DOCKER_IMAGE="${DOCKER_IMAGE:-cv-matcher:latest}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"

# Functions
print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

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
    echo -e "${BLUE}ℹ $1${NC}"
}

# Deploy vLLM
deploy_vllm() {
    print_header "Deploying vLLM Server"
    
    if kubectl apply -f k8s/vllm-deployment.yaml; then
        print_success "vLLM deployment applied"
    else
        print_error "Failed to apply vLLM deployment"
        return 1
    fi
    
    print_info "Waiting for vLLM pod to be ready (this may take several minutes)..."
    if kubectl wait --for=condition=ready pod -l app=vllm-server --timeout=600s -n $NAMESPACE; then
        print_success "vLLM pod is ready"
    else
        print_error "vLLM pod failed to become ready"
        kubectl logs -l app=vllm-server -n $NAMESPACE
        return 1
    fi
}

# Deploy CV Matcher Job
deploy_job() {
    print_header "Deploying CV Matcher Job"
    
    if kubectl apply -f k8s/cv-matcher-job.yaml; then
        print_success "CV Matcher job applied"
    else
        print_error "Failed to apply CV Matcher job"
        return 1
    fi
    
    print_info "Job submitted. Monitoring execution..."
    sleep 2
}

# Monitor job execution
monitor_job() {
    print_header "Monitoring Job Execution"
    
    # Show job status
    kubectl describe job $CV_MATCHER_JOB -n $NAMESPACE
    
    # Follow logs
    print_info "Following job logs (Ctrl+C to stop)..."
    kubectl logs -f job/$CV_MATCHER_JOB -n $NAMESPACE || true
}

# Check deployment status
check_status() {
    print_header "Deployment Status"
    
    print_info "vLLM Deployment:"
    kubectl get deployment $VLLM_DEPLOYMENT -n $NAMESPACE
    kubectl get pods -l app=vllm-server -n $NAMESPACE
    
    echo ""
    print_info "CV Matcher Job:"
    kubectl get job $CV_MATCHER_JOB -n $NAMESPACE
    kubectl get pods -l job-name=$CV_MATCHER_JOB -n $NAMESPACE
    
    echo ""
    print_info "Services:"
    kubectl get svc -l app=vllm-server -n $NAMESPACE
    
    echo ""
    print_info "Persistent Volumes:"
    kubectl get pvc -n $NAMESPACE
}

# Test vLLM connectivity
test_vllm() {
    print_header "Testing vLLM Connectivity"
    
    # Port-forward to vLLM service
    print_info "Port-forwarding vLLM service..."
    kubectl port-forward svc/vllm-service 8000:8000 -n $NAMESPACE > /dev/null 2>&1 &
    PF_PID=$!
    
    # Wait for port-forward to establish
    sleep 2
    
    # Test models endpoint
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        print_success "vLLM is responding to API requests"
        curl -s http://localhost:8000/v1/models | jq .
    else
        print_error "vLLM is not responding"
    fi
    
    # Cleanup port-forward
    kill $PF_PID 2>/dev/null || true
}

# View job logs
view_logs() {
    print_header "Job Logs"
    kubectl logs -f job/$CV_MATCHER_JOB -n $NAMESPACE
}

# Clean up resources
cleanup() {
    print_header "Cleaning Up Resources"
    
    read -p "Are you sure you want to delete vLLM deployment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete deployment $VLLM_DEPLOYMENT -n $NAMESPACE || true
        kubectl delete service vllm-service -n $NAMESPACE || true
        print_success "vLLM deployment deleted"
    fi
    
    read -p "Are you sure you want to delete CV Matcher job? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete job $CV_MATCHER_JOB -n $NAMESPACE || true
        print_success "CV Matcher job deleted"
    fi
    
    read -p "Are you sure you want to delete PVC? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete pvc cv-matcher-output-pvc -n $NAMESPACE || true
        print_success "PVC deleted"
    fi
}

# Retrieve results
retrieve_results() {
    print_header "Retrieving Results"
    
    # Get pod name
    POD_NAME=$(kubectl get pod -l job-name=$CV_MATCHER_JOB -o jsonpath='{.items[0].metadata.name}' -n $NAMESPACE)
    
    if [ -z "$POD_NAME" ]; then
        print_error "No pod found for job"
        return 1
    fi
    
    print_info "Pod: $POD_NAME"
    
    # Copy CSV from pod
    LOCAL_DIR="./results-$(date +%Y%m%d-%H%M%S)"
    mkdir -p $LOCAL_DIR
    
    kubectl cp $NAMESPACE/$POD_NAME:/data/output/ $LOCAL_DIR/ -c cv-matcher || true
    
    if [ -d "$LOCAL_DIR" ] && [ "$(ls -A $LOCAL_DIR)" ]; then
        print_success "Results retrieved to: $LOCAL_DIR"
        ls -lah $LOCAL_DIR
    else
        print_warning "No results found yet (job may still be running)"
    fi
}

# Show usage
usage() {
    cat << EOF
CV Matcher Kubernetes Deployment Helper

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  deploy-all          Deploy both vLLM and CV Matcher job
  deploy-vllm         Deploy only vLLM server
  deploy-job          Deploy only CV Matcher job
  status              Show deployment status
  logs                Follow job logs
  test-vllm           Test vLLM connectivity
  monitor             Monitor job execution and follow logs
  retrieve            Retrieve results from completed job
  cleanup             Delete all deployed resources
  help                Show this help message

Environment Variables:
  NAMESPACE           Kubernetes namespace (default: default)
  DOCKER_IMAGE        Docker image for CV Matcher (default: cv-matcher:latest)
  DOCKER_REGISTRY     Docker registry prefix

Examples:
  # Deploy everything
  $0 deploy-all
  
  # Check status
  $0 status
  
  # View logs
  $0 logs
  
  # Retrieve results
  $0 retrieve

EOF
}

# Main
main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi
    
    case "$1" in
        deploy-all)
            deploy_vllm && deploy_job && print_success "All resources deployed"
            ;;
        deploy-vllm)
            deploy_vllm
            ;;
        deploy-job)
            deploy_job
            ;;
        status)
            check_status
            ;;
        logs)
            view_logs
            ;;
        test-vllm)
            test_vllm
            ;;
        monitor)
            monitor_job
            ;;
        retrieve)
            retrieve_results
            ;;
        cleanup)
            cleanup
            ;;
        help)
            usage
            ;;
        *)
            print_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
