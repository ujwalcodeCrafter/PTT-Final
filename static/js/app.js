document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the index page for upload functionality
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('upload-preview');
    const placeholderSection = document.getElementById('upload-placeholder');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const progressSection = document.getElementById('upload-progress');
    const progressBar = document.getElementById('upload-progress-bar');

    let selectedFile = null;

    // Auto-trigger analysis if coming from gallery with an image
    const galleryImageUrl = sessionStorage.getItem('gallery_image_url');
    if (galleryImageUrl) {
        sessionStorage.removeItem('gallery_image_url');
        // Fetch the image and set it as the selected file
        fetch(galleryImageUrl)
            .then(r => r.blob())
            .then(blob => {
                const file = new File([blob], 'gallery_image.jpg', { type: blob.type });
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('preview-image').src = e.target.result;
                    placeholderSection.classList.add('d-none');
                    previewSection.classList.remove('d-none');
                    // Auto-start analysis
                    analyzeBtn.click();
                };
                reader.readAsDataURL(file);
            })
            .catch(err => {
                console.error('Failed to load gallery image:', err);
            });
    }

    // Upload zone click handler
    if (uploadZone) {
        uploadZone.addEventListener('click', (e) => {
            if (e.target !== analyzeBtn && e.target !== resetBtn) {
                fileInput.click();
            }
        });

        // Drag and drop handlers
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('drag-over');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('drag-over');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });

        // File input handler
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('preview-image').src = e.target.result;
            placeholderSection.classList.add('d-none');
            previewSection.classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    }

    // Analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            const formData = new FormData();
            formData.append('image', selectedFile);

            // Show loading state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
            progressSection.classList.remove('d-none');
            progressBar.style.width = '30%';

            try {
                progressBar.style.width = '60%';

                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData,
                });

                const result = await response.json();

                progressBar.style.width = '100%';

                if (result.error) {
                    showToast('Error', result.error, 'danger');
                    resetUploadForm();
                } else {
                    setTimeout(() => {
                        displayResults(result);
                        showToast('Success', 'Analysis complete!', 'success');
                    }, 500);
                }
            } catch (error) {
                showToast('Error', 'Connection failed. Please try again.', 'danger');
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="bi bi-search me-1"></i>Analyze Image';
                progressSection.classList.add('d-none');
            }
        });
    }

    // Reset button
    if (resetBtn) {
        resetBtn.addEventListener('click', resetUploadForm);
    }

    function resetUploadForm() {
        selectedFile = null;
        fileInput.value = '';
        document.getElementById('preview-image').src = '';
        previewSection.classList.add('d-none');
        placeholderSection.classList.remove('d-none');
        progressSection.classList.add('d-none');
        document.getElementById('result-section').classList.add('d-none');
        // Explicitly clear heatmap to prevent stale image flash
        const heatmapContainer = document.getElementById('heatmap-container');
        if (heatmapContainer) {
            heatmapContainer.innerHTML = '';
        }
    }

    function displayResults(result) {
        const resultSection = document.getElementById('result-section');
        resultSection.classList.remove('d-none');
        resultSection.classList.add('result-animate');
        resultSection.scrollIntoView({ behavior: 'smooth' });

        // Verdict card
        const { fraud_probability, risk_level, is_fraud } = result;
        const probPercentage = Math.round(fraud_probability * 100);

        updateGauge(probPercentage);
        updateVerdictCard(is_fraud, risk_level);

        // Risk level text
        const riskLevelEl = document.getElementById('risk-level');
        const levelMap = { low: 'Low', medium: 'Medium', high: 'High' };
        riskLevelEl.textContent = `Risk Level: ${levelMap[risk_level] || 'Unknown'}`;

        // Report
        if (result.report) {
            const reportEl = document.getElementById('report-content');
            reportEl.innerHTML = '';
            if (result.report.top_suspicious_features && result.report.top_suspicious_features.length > 0) {
                result.report.top_suspicious_features.forEach((feature, i) => {
                    reportEl.innerHTML += `
                        <div class="feature-item">
                            <span><small>${feature.name || `Feature ${i}`}</small></span>
                            <span class="badge bg-warning">${(feature.importance * 100).toFixed(1)}%</span>
                        </div>`;
                });
            } else {
                reportEl.innerHTML = '<p class="text-muted text-center">No suspicious features detected</p>';
            }

            // Add verdict text
            const verdictText = result.report.verdict || 'Analysis complete';
            reportEl.innerHTML = `
                <div class="alert ${is_fraud ? 'alert-danger' : 'alert-success'} mb-3">
                    <strong>Verdict:</strong> ${verdictText}
                </div>` + reportEl.innerHTML;

            // Add classification
            const classif = result.report.classification;
            if (classif) {
                reportEl.innerHTML += `
                    <div class="feature-item mt-2">
                        <span><small>Classification</small></span>
                        <span class="badge ${is_fraud ? 'bg-danger' : 'bg-success'}">${classif}</span>
                    </div>`;
            }
        }

        // Heatmap
        const heatmapContainer = document.getElementById('heatmap-container');
        if (result.heatmap_url) {
            heatmapContainer.innerHTML = `
                <img src="${result.heatmap_url}" alt="Heatmap" class="img-fluid">
            `;
        } else {
            heatmapContainer.innerHTML = `
                <div class="text-muted py-3">
                    <i class="bi bi-image-alt display-4"></i>
                    <p>No heatmap available</p>
                </div>`;
        }
    }

    function updateGauge(percent) {
        const gaugeFill = document.getElementById('gauge-fill');
        const gaugeText = document.getElementById('gauge-text');
        if (gaugeFill && gaugeText) {
            const circumference = 2 * Math.PI * 90; // ~565.48
            const offset = circumference - (percent / 100) * circumference;
            gaugeFill.style.strokeDashoffset = offset;
            gaugeText.textContent = `${percent}%`;

            // Color based on risk
            let color = '#28a745';
            if (percent > 60) {
                color = '#dc3545';
            } else if (percent > 35) {
                color = '#ffc107';
            }
            gaugeFill.style.stroke = color;
        }
    }

    function updateVerdictCard(isFraud, riskLevel) {
        const header = document.getElementById('verdict-header');
        const title = document.getElementById('verdict-title');

        header.classList.remove('bg-success', 'bg-warning', 'bg-danger');

        if (isFraud) {
            header.classList.add('bg-danger');
            title.innerHTML = '<i class="bi bi-x-circle me-2"></i>Fraud Detected';
        } else if (riskLevel === 'medium') {
            header.classList.add('bg-warning');
            title.innerHTML = '<i class="bi bi-exclamation-triangle me-2"></i>Possible Fraud';
        } else {
            header.classList.add('bg-success');
            title.innerHTML = '<i class="bi bi-check-circle me-2"></i>Genuine';
        }
    }

    function showToast(title, message, type) {
        const container = document.getElementById('toast-container');
        const id = 'toast-' + Date.now();
        const icons = {
            success: 'bi-check-circle',
            danger: 'bi-exclamation-circle',
            warning: 'bi-exclamation-triangle',
            info: 'bi-info-circle',
        };

        const toastHTML = `
            <div id="${id}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="bi ${icons[type] || icons.info} me-2"></i><strong>${title}</strong><br>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-${type} me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>`;

        container.insertAdjacentHTML('beforeend', toastHTML);
        const toastEl = document.getElementById(id);
        const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    }

    // Health check
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            console.log('Health check:', data);
        })
        .catch(err => console.warn('Health check failed:', err));
});
