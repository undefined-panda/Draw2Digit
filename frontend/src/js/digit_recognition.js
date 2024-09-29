let isDrawing = false;
let p5Canvas;

function setup() {
    // Nutze die Canvas-ID, um das Canvas zu selektieren
    let canvasElement = document.getElementById('canvasPosition');
    p5Canvas = createCanvas(canvasElement.height, canvasElement.width);
    // Platziere das p5.js-Canvas genau über dem HTML-Canvas
    p5Canvas.class("canvasDrawing")
    p5Canvas.parent(canvasElement.parentNode);
    // Lösche das automatisch erstellte p5.js-Canvas
    canvasElement.remove();
    background(0);  // Setze den Hintergrund auf schwarz
}

// function to draw with mouse
function draw() {
    if (isDrawing) {
        stroke(255);
        strokeWeight(20);
        line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function mousePressed() {
    isDrawing = true;
}

function mouseReleased() {
    isDrawing = false;
}

// function to clear the drawing area
function clearArea() {
    clear()
    background(0);
}

function getGrayscale(imageData, height, width) {
    // pixelvalues are in the RDGBA order, order goes by rows from the top-left pixel to the bottom-right.
    let grayscaleImage = [];
    for (let y=0; y<height; y++){
        let row = [];
        for (let x=0; x<width; x++){
            const index = (x + y * width) * 4;
            const r = imageData[index];
            const g = imageData[index + 1];
            const b = imageData[index + 2];
            // convert to grayscale
            const gray = (r + g + b) / 3;
            row.push(gray);
        }
        grayscaleImage.push(row);
    }

  return grayscaleImage;
}

// function to send the drawn image to the backend for further processing
function sendImage() {
    const nativeCanvas = p5Canvas.elt; // get native canvas
    const ctx = nativeCanvas.getContext('2d'); 

    const height = nativeCanvas.height;
    const width = nativeCanvas.width;
    const imageData = ctx.getImageData(0, 0, width, height).data; // get the drawing

    let grayscaleImage = getGrayscale(imageData, height, width)

    fetch('/pages/digit_recognition', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ drawing: grayscaleImage }),
        }).then(response => {
            if (response.ok) {
                console.log('Bild erfolgreich gesendet!');
                return response.json(); 
            } else {
                throw new Error('Fehler beim Senden des Bildes.');
            }
        }).then(data => {
            const prediction = data.processed_value
            document.getElementById("processedValue").innerHTML = prediction
            
            const imgSrc = `data:image/png;base64,${data.image}`;
            document.getElementById('resizedImage').src = imgSrc;
            console.log(data)
        }).catch(error => {
            console.error('Fehler:', error);  // Behandle Fehler
        });
}