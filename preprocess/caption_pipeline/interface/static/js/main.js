/**
 * Main JavaScript functionality for the Arabic video dataset interface
 */

// Toast notification system
class Toast {
    /**
     * Creates and shows a toast notification
     * @param {string} message - The message to display
     * @param {string} type - The type of toast (success, warning, danger, info)
     * @param {number} duration - How long to display the toast in milliseconds
     */
    static show(message, type = 'success', duration = 3000) {
        const container = document.querySelector('.toast-container');
        if (!container) return;
        
        const iconClass = type === 'success' ? 'fa-check-circle' : 
                         type === 'warning' ? 'fa-exclamation-triangle' : 
                         type === 'danger' ? 'fa-exclamation-circle' : 
                         type === 'info' ? 'fa-info-circle' : 'fa-bell';
        
        const toast = document.createElement('div');
        toast.className = `toast show align-items-center text-white bg-${type} border-0 mb-2`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas ${iconClass} me-2" aria-hidden="true"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        container.appendChild(toast);
        
        // Remove the toast after duration
        setTimeout(() => {
            toast.classList.add('hide');
            setTimeout(() => {
                container.removeChild(toast);
            }, 300); // Wait for animation to finish
        }, duration);
        
        // Add click handler to close button
        const closeBtn = toast.querySelector('.btn-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                toast.classList.add('hide');
                setTimeout(() => {
                    try {
                        container.removeChild(toast);
                    } catch (e) {
                        // Toast may have already been removed
                    }
                }, 300);
            });
        }
    }
}

/**
 * Date utilities for handling UTC to local time conversion
 */
class DateUtils {
    /**
     * Convert a UTC date string to local timezone and format it
     * @param {string} utcDateString - The UTC date string to convert (YYYY-MM-DD HH:MM)
     * @param {string} format - Output format (optional)
     * @returns {string} - Formatted date string in local timezone
     */
    static convertUTCToLocal(utcDateString, format = 'YYYY-MM-DD HH:MM') {
        if (!utcDateString) return 'N/A';
        
        let date;
        
        // Check if the string is in ISO format
        if (utcDateString.includes('T') && (utcDateString.includes('Z') || utcDateString.includes('+'))) {
            // Handle ISO format directly
            date = new Date(utcDateString);
        } else {
            // Parse the UTC date string (assuming format YYYY-MM-DD HH:MM)
            const parts = utcDateString.split(' ');
            
            // If there's no time part, add default time
            if (parts.length === 1) {
                parts.push('00:00');
            }
            
            const [datePart, timePart] = parts;
            const [year, month, day] = datePart.split('-').map(Number);
            const [hour, minute] = timePart.split(':').map(Number);
            
            // Create a UTC date object
            date = new Date(Date.UTC(year, month - 1, day, hour, minute));
        }
        
        // Get local date components
        const localYear = date.getFullYear();
        const localMonth = date.getMonth() + 1; // getMonth() returns 0-11
        const localDay = date.getDate();
        const localHour = date.getHours();
        const localMinute = date.getMinutes();
        
        // Format based on requested format
        let formattedDate;
        if (format === 'YYYY-MM-DD HH:MM') {
            formattedDate = `${localYear}-${localMonth.toString().padStart(2, '0')}-${localDay.toString().padStart(2, '0')} ${localHour.toString().padStart(2, '0')}:${localMinute.toString().padStart(2, '0')}`;
        } else if (format === 'YYYY-MM-DD') {
            formattedDate = `${localYear}-${localMonth.toString().padStart(2, '0')}-${localDay.toString().padStart(2, '0')}`;
        } else if (format === 'HH:MM') {
            formattedDate = `${localHour.toString().padStart(2, '0')}:${localMinute.toString().padStart(2, '0')}`;
        } else if (format === 'MM/DD/YYYY') {
            formattedDate = `${localMonth.toString().padStart(2, '0')}/${localDay.toString().padStart(2, '0')}/${localYear}`;
        } else if (format === 'readable') {
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            formattedDate = `${months[localMonth - 1]} ${localDay}, ${localYear} at ${localHour.toString().padStart(2, '0')}:${localMinute.toString().padStart(2, '0')}`;
        } else if (format === 'YYYY') {
            formattedDate = `${localYear}`;
        }
        
        return formattedDate;
    }
    
    /**
     * Initialize date conversions for elements with data-utc-date attribute
     */
    static initDateConversions() {
        // Convert all elements with data-utc-date attribute
        document.querySelectorAll('[data-utc-date]').forEach(el => {
            const utcDate = el.getAttribute('data-utc-date');
            const format = el.getAttribute('data-date-format') || 'YYYY-MM-DD HH:MM';
            el.textContent = DateUtils.convertUTCToLocal(utcDate, format);
        });
    }
}

/**
 * Auto-expanding textarea functionality
 */
function setupAutoExpandTextarea() {
    const textareas = document.querySelectorAll('.auto-expand-textarea[data-auto-expand="true"]');
    
    textareas.forEach(textarea => {
        // Simple auto-expand function
        function autoExpand() {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        }
        
        // Set initial height
        autoExpand();
        
        // Add event listeners
        textarea.addEventListener('input', autoExpand);
        textarea.addEventListener('paste', () => {
            setTimeout(autoExpand, 10);
        });
    });
}

// Make the simple function available globally
window.adjustTextareaHeight = function(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
};

document.addEventListener('DOMContentLoaded', function() {
    // Enable tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            trigger: 'hover focus'
        });
    });
    
    // Enable popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize date timezone conversions
    DateUtils.initDateConversions();
    
    // Convert old-style confirmation dialogs to modern modals
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const message = this.getAttribute('data-confirm') || 'Are you sure you want to proceed?';
            const title = this.getAttribute('data-confirm-title') || 'Confirmation';
            const confirmText = this.getAttribute('data-confirm-text') || 'Confirm';
            const cancelText = this.getAttribute('data-confirm-cancel') || 'Cancel';
            const href = this.getAttribute('href') || this.form?.action;
            const isForm = this.form != null;
            
            // Create modal dynamically
            const modalId = 'confirm-modal-' + Math.random().toString(36).substring(2);
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.id = modalId;
            modal.setAttribute('tabindex', '-1');
            modal.setAttribute('aria-hidden', 'true');
            
            modal.innerHTML = `
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>${message}</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">${cancelText}</button>
                            <button type="button" class="btn btn-primary confirm-action">${confirmText}</button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const modalElement = new bootstrap.Modal(modal);
            modalElement.show();
            
            // Handle confirm action
            modal.querySelector('.confirm-action').addEventListener('click', function() {
                modalElement.hide();
                
                // If it's a form submit
                if (isForm) {
                    button.form.submit();
                } 
                // If it's a link
                else if (href) {
                    window.location.href = href;
                }
                
                // Remove modal after hiding
                modal.addEventListener('hidden.bs.modal', function() {
                    document.body.removeChild(modal);
                });
            });
            
            // Remove modal when hidden
            modal.addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        });
    });
    
    // Fix for RTL text in contenteditable elements
    const rtlEditable = document.querySelectorAll('[contenteditable][dir="rtl"]');
    rtlEditable.forEach(function(element) {
        // Set initial direction
        element.style.direction = 'rtl';
        element.style.textAlign = 'right';
        
        // Listen for input events
        element.addEventListener('input', function() {
            // Ensure RTL direction is maintained
            this.style.direction = 'rtl';
            this.style.textAlign = 'right';
        });
    });
    
    // Auto-close alerts with animation
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.classList.add('fade');
            setTimeout(function() {
                const closeButton = alert.querySelector('.btn-close');
                if (closeButton) {
                    closeButton.click();
                }
            }, 300);
        }, 5000);
    });
    
    // Handle AJAX forms
    const ajaxForms = document.querySelectorAll('form[data-ajax="true"]');
    ajaxForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitBtn = form.querySelector('[type="submit"]');
            const originalBtnText = submitBtn ? submitBtn.innerHTML : '';
            
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="loading-spinner me-2"></span> Processing...';
            }
            
            // Get form data
            const formData = new FormData(form);
            
            // Send AJAX request
            fetch(form.action, {
                method: form.method,
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    Toast.show(data.message || 'Operation completed successfully', 'success');
                    
                    // Handle assignment updates
                    if (data.assignment) {
                        const assignmentsList = document.querySelector('.list-group');
                        if (assignmentsList) {
                            const newAssignment = document.createElement('div');
                            newAssignment.className = 'list-group-item d-flex justify-content-between align-items-center';
                            newAssignment.innerHTML = `
                                <div class="form-check d-flex align-items-center">
                                    <input class="form-check-input assignment-select" type="checkbox" 
                                           value="${data.assignment.user_id}" id="assignment-${data.assignment.user_id}">
                                    <label class="form-check-label ms-2" for="assignment-${data.assignment.user_id}">
                                        <div>
                                            <h6 class="mb-0">${data.assignment.username}</h6>
                                            <small class="text-muted">Assigned ${data.assignment.assigned_at}</small>
                                        </div>
                                    </label>
                                </div>
                                <form action="/admin/unassign-video/${form.dataset.videoId}" method="POST" class="d-inline">
                                    <input type="hidden" name="user_id" value="${data.assignment.user_id}">
                                    <button type="submit" class="btn btn-outline-danger btn-sm" 
                                            onclick="return confirm('Are you sure you want to unassign ${data.assignment.username}?')">
                                        <i class="fas fa-times"></i>
                                    </button>
                                </form>
                            `;
                            assignmentsList.appendChild(newAssignment);
                            
                            // Add event listener to the new checkbox
                            const newCheckbox = newAssignment.querySelector('.assignment-select');
                            if (newCheckbox) {
                                newCheckbox.addEventListener('change', function() {
                                    // Find and trigger the updateSelectedState function
                                    const event = new Event('change');
                                    document.dispatchEvent(new CustomEvent('update-selected-state'));
                                });
                            }
                            
                            // Clear the select dropdown
                            const selectElement = form.querySelector('select');
                            if (selectElement) {
                                selectElement.value = '';
                            }
                            
                            // Close the modal if it exists
                            const modal = bootstrap.Modal.getInstance(form.closest('.modal'));
                            if (modal) {
                                modal.hide();
                            }
                        }
                    }
                    
                    // If callback function is specified, call it
                    if (form.dataset.callback) {
                        const callback = window[form.dataset.callback];
                        if (typeof callback === 'function') {
                            callback(data);
                        }
                    }
                } else {
                    Toast.show(data.message || 'An error occurred', 'danger');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                Toast.show('An unexpected error occurred', 'danger');
            })
            .finally(() => {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalBtnText;
                }
            });
        });
    });
    
    // Initialize auto-expanding textareas
    setupAutoExpandTextarea();
});

// Detect inactive sessions
(function() {
    let inactivityTime = 0;
    const inactivityLimit = 60 * 60; // 60 minutes in seconds (increased from 30)
    let inactivityInterval;
    let warningShown = false;
    
    function resetInactivityTimer() {
        inactivityTime = 0;
        if (warningShown) {
            warningShown = false;
            // Hide any existing inactivity warnings
            // ...
        }
    }
    
    function setupInactivityDetection() {
        // Reset the timer on user activity
        ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetInactivityTimer, false);
        });
        
        // Check inactivity every second
        inactivityInterval = setInterval(() => {
            inactivityTime++;
            
            // Show warning at 50 minutes (10 minutes before timeout)
            if (inactivityTime >= (inactivityLimit - 600) && !warningShown) {
                warningShown = true;
                Toast.show('Your session will expire soon due to inactivity. Please continue working to stay logged in.', 'warning', 10000);
            }
            
            // Logout after inactivity limit
            if (inactivityTime >= inactivityLimit) {
                clearInterval(inactivityInterval);
                Toast.show('Your session has expired due to inactivity. Please log in again.', 'info');
                setTimeout(() => {
                    window.location.href = '/auth/logout';
                }, 3000);
            }
        }, 1000);
    }
    
    // Only setup inactivity detection for logged-in users
    if (document.querySelector('.navbar-nav .dropdown-toggle')) {
        setupInactivityDetection();
    }
})();

// Make Toast available globally
window.Toast = Toast;
