document.addEventListener('DOMContentLoaded', () => {
    const filterButtons = document.querySelectorAll('.filters button');
    const reportSections = document.querySelectorAll('#report-content section');
    const searchInput = document.getElementById('search-input');
    let activeFilter = 'all';
    let currentHighlightTerm = '';

    // --- Filtering Logic --- 
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            activeFilter = button.getAttribute('data-filter');
            
            // Update active button style
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            applyFiltersAndSearch();
        });
    });

    // --- Search Logic --- 
    let searchTimeout;
    searchInput.addEventListener('input', () => {
        // Debounce search input
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            currentHighlightTerm = searchInput.value.trim();
            applyFiltersAndSearch();
        }, 300); // Adjust delay as needed (300ms)
    });

    function applyFiltersAndSearch() {
        const searchTerm = currentHighlightTerm.toLowerCase();
        
        reportSections.forEach(section => {
            const category = section.getAttribute('data-category');
            const sectionText = section.textContent.toLowerCase();
            
            // 1. Apply Category Filter
            const categoryMatch = (activeFilter === 'all' || category === activeFilter);
            
            // 2. Apply Search Term Filter (only if search term exists)
            const searchMatch = (searchTerm === '' || sectionText.includes(searchTerm));
            
            // Show/Hide Section based on both filters
            if (categoryMatch && searchMatch) {
                section.classList.remove('hidden');
            } else {
                section.classList.add('hidden');
            }
            
            // 3. Apply Highlighting (only if search term exists and section is visible)
            if (searchTerm !== '' && categoryMatch && searchMatch) {
                highlightTextInSection(section, searchTerm);
            } else {
                removeHighlightFromSection(section);
            }
        });
    }

    // --- Highlighting Helper Functions ---
    function highlightTextInSection(section, term) {
        // Remove previous highlights
        removeHighlightFromSection(section);

        if (!term) return;

        const walker = document.createTreeWalker(section, NodeFilter.SHOW_TEXT, null, false);
        let node;
        const termRegex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        const nodesToModify = [];

        while (node = walker.nextNode()) {
            if (node.nodeValue.toLowerCase().includes(term)) {
                nodesToModify.push(node);
            }
        }

        nodesToModify.forEach(textNode => {
            const parent = textNode.parentNode;
            const parts = textNode.nodeValue.split(termRegex);
            
            parts.forEach((part, index) => {
                if (index % 2 === 1) { // This is the matching term
                    const span = document.createElement('span');
                    span.className = 'highlight';
                    span.textContent = part;
                    parent.insertBefore(span, textNode);
                } else if (part.length > 0) {
                    parent.insertBefore(document.createTextNode(part), textNode);
                }
            });
            parent.removeChild(textNode);
        });
    }

    function removeHighlightFromSection(section) {
        const highlights = section.querySelectorAll('span.highlight');
        highlights.forEach(span => {
            const parent = span.parentNode;
            parent.replaceChild(document.createTextNode(span.textContent), span);
            parent.normalize(); // Merge adjacent text nodes
        });
    }

    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
    }
    
    // --- Initial Setup --- 
    // Apply initial filter (show all)
    applyFiltersAndSearch(); 
    
    // --- Optional: Markdown Rendering --- 
    // Uncomment if you add a Markdown library like Marked.js
    /*
    const rawMarkdownElement = document.getElementById('raw-markdown');
    const renderedContentElement = document.getElementById('markdown-rendered-content');
    if (rawMarkdownElement && renderedContentElement && typeof marked !== 'undefined') {
        try {
            renderedContentElement.innerHTML = marked.parse(rawMarkdownElement.textContent);
        } catch (error) {
            console.error('Error parsing Markdown:', error);
            renderedContentElement.innerHTML = '<p class="error">Error rendering Markdown.</p>';
        }
    } else if (rawMarkdownElement) {
         renderedContentElement.innerHTML = '<p class="error">Markdown library (e.g., Marked.js) not loaded.</p>';
    }
    */
}); 