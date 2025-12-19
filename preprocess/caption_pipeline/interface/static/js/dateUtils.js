/**
 * Utility functions for date/time handling
 */
const DateUtils = {
    /**
     * Formats a date string according to the specified format or using locale-based formatting
     * @param {string} dateString - ISO format date string
     * @param {string} format - Simple format string (YYYY-MM-DD HH:mm)
     * @return {string} Formatted date string
     */
    formatDate: function(dateString, format) {
        if (!dateString) return '';
        
        try {
            const date = new Date(dateString);
            
            if (isNaN(date.getTime())) {
                return dateString; // Return original if invalid
            }
            
            // If format is provided, do manual formatting
            if (format) {
                // Basic formatting replacements - note we use 'mm' for minutes, not 'MM'
                const replacements = {
                    'YYYY': date.getFullYear(),
                    'MM': String(date.getMonth() + 1).padStart(2, '0'),
                    'DD': String(date.getDate()).padStart(2, '0'),
                    'HH': String(date.getHours()).padStart(2, '0'),
                    'mm': String(date.getMinutes()).padStart(2, '0'),
                    'ss': String(date.getSeconds()).padStart(2, '0')
                };
                
                // Replace format tokens with actual values
                let result = format;
                for (const [key, value] of Object.entries(replacements)) {
                    result = result.replace(key, value);
                }
                
                return result;
            }
            
            // Otherwise use browser's locale formatting
            return this.formatDateLocale(date);
        } catch (e) {
            console.error('Error formatting date:', e);
            return dateString;
        }
    },
    
    /**
     * Format date using browser's locale
     * @param {Date} date - The date object to format
     * @return {string} Localized date string
     */
    formatDateLocale: function(date) {
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        
        return date.toLocaleDateString(navigator.language || 'en', options);
    },
    
    /**
     * Format just the date part using browser's locale
     * @param {Date} date - The date object to format
     * @return {string} Localized date string without time
     */
    formatDateOnlyLocale: function(date) {
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric'
        };
        
        return date.toLocaleDateString(navigator.language || 'en', options);
    },
    
    /**
     * Convert UTC dates to local dates for elements with data-utc-date attributes
     */
    initDateConversions: function() {
        document.querySelectorAll('[data-utc-date]').forEach(element => {
            const utcDate = element.getAttribute('data-utc-date');
            if (!utcDate) return;
            
            const date = new Date(utcDate);
            if (isNaN(date.getTime())) return;
            
            const format = element.getAttribute('data-date-format');
            
            // Use format if provided, otherwise use locale formatting
            if (format) {
                element.textContent = this.formatDate(utcDate, format);
            } else {
                // If format contains "date-only", only show the date part
                if (element.hasAttribute('data-date-only')) {
                    element.textContent = this.formatDateOnlyLocale(date);
                } else {
                    element.textContent = this.formatDateLocale(date);
                }
            }
        });
    }
};

// Initialize date conversions when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    DateUtils.initDateConversions();
});