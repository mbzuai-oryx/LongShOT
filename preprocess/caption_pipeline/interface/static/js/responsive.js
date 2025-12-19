/**
 * responsive.js - Handle responsive behaviors and mobile optimizations
 */

document.addEventListener('DOMContentLoaded', function() {
    // Detect if user is on a mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Add a class to the body for mobile-specific styling
    if (isMobile) {
        document.body.classList.add('is-mobile-device');
    }
    
    // Handle video player sizing on mobile
    const videoPlayer = document.getElementById('video-player');
    if (videoPlayer && isMobile) {
        // Ensure video controls are enabled
        videoPlayer.controls = true;
        // Add mobile optimized controls
        videoPlayer.setAttribute('playsinline', '');
        videoPlayer.setAttribute('webkit-playsinline', '');
    }
    
    // Optimize UI for small screens
    if (window.innerWidth < 768) {
        // Collapse sidebar sections by default on mobile
        const collapsibleSections = document.querySelectorAll('.sidebar-section[data-collapsible="true"]');
        collapsibleSections.forEach(section => {
            const content = section.querySelector('.sidebar-section-content');
            const heading = section.querySelector('h5, h6');
            
            if (content && heading) {
                // Add toggle indicator
                heading.innerHTML += ' <i class="fas fa-chevron-down toggle-indicator"></i>';
                heading.style.cursor = 'pointer';
                
                // Initially collapse on mobile
                content.style.display = 'none';
                
                // Toggle functionality
                heading.addEventListener('click', function() {
                    const isVisible = content.style.display !== 'none';
                    content.style.display = isVisible ? 'none' : 'block';
                    
                    // Update indicator
                    const indicator = heading.querySelector('.toggle-indicator');
                    if (indicator) {
                        indicator.className = isVisible 
                            ? 'fas fa-chevron-down toggle-indicator' 
                            : 'fas fa-chevron-up toggle-indicator';
                    }
                });
            }
        });
    }
    
    // Handle orientation changes
    window.addEventListener('orientationchange', function() {
        // Adjust UI elements after orientation change
        setTimeout(function() {
            // Refresh any components that need adjusting
            const videoContainer = document.querySelector('.ratio');
            if (videoContainer) {
                // Force redraw of responsive containers
                videoContainer.style.opacity = '0.99';
                setTimeout(() => {
                    videoContainer.style.opacity = '1';
                }, 50);
            }
        }, 200);
    });
    
    // Touch-friendly adjustments
    if ('ontouchstart' in window) {
        // Make clickable elements larger on touch devices
        document.querySelectorAll('.btn-sm').forEach(btn => {
            btn.classList.remove('btn-sm');
        });
        
        // Add touch feedback effect
        document.querySelectorAll('.btn, .card, .nav-link, .page-link').forEach(element => {
            element.addEventListener('touchstart', function() {
                this.classList.add('touch-active');
            });
            
            element.addEventListener('touchend', function() {
                this.classList.remove('touch-active');
            });
        });
    }
});

// Add resize handler for responsive adjustments
window.addEventListener('resize', function() {
    // Simple resize handling for textareas
    const autoTextareas = document.querySelectorAll('.auto-expand-textarea[data-auto-expand="true"]');
    autoTextareas.forEach(textarea => {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    });
    
    // Any dynamic layout adjustments needed on resize
    const captionList = document.getElementById('caption-list');
    if (captionList) {
        // Adjust height based on available space
        const header = document.querySelector('.card-header');
        const footer = document.querySelector('.card-footer');
        if (header && footer) {
            const availableHeight = window.innerHeight - header.offsetHeight - footer.offsetHeight - 200;
            captionList.style.maxHeight = Math.max(300, availableHeight) + 'px';
        }
    }
});
