document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const prediction = document.getElementById('prediction');
    const loading = document.getElementById('loading');

    // Handle file input change
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                predictBtn.disabled = false;
                prediction.textContent = '';
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.innerHTML = '';
            predictBtn.disabled = true;
        }
    });

    // Handle predict button click
    predictBtn.addEventListener('click', predict);
    predictBtn.disabled = true; // Initially disabled

    async function predict() {
        const img = imageInput.files[0];
        if (!img) {
            alert("Please upload an image first");
            return;
        }

        // Show loading
        loading.style.display = 'flex';
        prediction.textContent = '';
        predictBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', img);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Prediction result:', result);

            // Display main prediction
            let predictionText = `Predicted: ${result.prediction}`;
            
            // Show debug info if available
            if (result.class_index !== undefined) {
                predictionText += ` (Index: ${result.class_index})`;
            }
            if (result.confidence !== undefined) {
                const confidencePercent = (result.confidence * 100).toFixed(1);
                predictionText += ` (${confidencePercent}% confidence)`;
            }

            prediction.textContent = predictionText;

        } catch (err) {
            console.error('Error:', err);
            prediction.textContent = 'Error: Unable to predict. Please try again.';
        } finally {
            // Hide loading
            loading.style.display = 'none';
            predictBtn.disabled = false;
        }
    }
});