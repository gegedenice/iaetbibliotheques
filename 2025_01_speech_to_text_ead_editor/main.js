import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

// Initialize Ace Editor
const editor = ace.edit("editor");
editor.setTheme("ace/theme/monokai");
editor.session.setMode("ace/mode/xml");
//editor.setValue('<?xml version="1.0" encoding="UTF-8"?>\n<ead>\n  <!-- EAD content will appear here -->\n</ead>');
//editor.setValue('<?xml version="1.0" encoding="UTF-8"?>\n');
editor.container.addEventListener('contextmenu', showContextMenu);

// Initialize Whisper pipeline
let whisperPipeline;
let mediaRecorder;
let audioChunks = [];

async function initWhisper() {
  // Show the loading spinner
  $('#loadingSpinner').show();
  try {
    whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small',
      {
        device: "webgpu",
        dtype: 'fp32'
      },);
    // Hide the loading spinner after the model is loaded
    $('#loadingSpinner').hide();
    $('#status').text('Ready to record');
  } catch (e) {
    $('#status').text('Error initializing Whisper: ' + e.message);
    $('#loadingSpinner').hide();
  }
}

// Initialize recording functionality
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      await processAudio(audioBlob);
    };

    mediaRecorder.start();
    $('#startRecording').prop('disabled', true);
    $('#stopRecording').prop('disabled', false);
    $('#status').text('Recording...');
  } catch (e) {
    $('#status').text('Error starting recording: ' + e.message);
  }
}

async function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    $('#startRecording').prop('disabled', false);
    $('#stopRecording').prop('disabled', true);
    $('#status').text('Processing audio...');
  }
}

async function processAudio(audioBlob) {
  $('#loadingSpinner').show();
  try {
    // Create a URL for the audio blob
    const audioUrl = URL.createObjectURL(audioBlob);

    // Create a download link for the audio file (optional)
    /*const downloadLink = document.createElement('a');
    downloadLink.href = audioUrl;
    downloadLink.download = 'recording.wav';
    downloadLink.textContent = 'Download audio file';
    document.body.appendChild(downloadLink);*/
    // Pass the Float32Array to the whisperPipeline
    const transcription = await whisperPipeline(audioUrl);
    $('#transcription').val(transcription.text);

    // Clean up the URL object
    URL.revokeObjectURL(audioUrl);
	$('#loadingSpinner').hide();
  } catch (e) {
	$('#loadingSpinner').hide();
    $('#status').text('Error processing audio: ' + e.message);
  }
}

// Function to send the final user prompt to the Ollama model
async function sendPrompt() {
  $('#loadingSpinner').show(); // Show the loading spinner
  const transcription = $('#transcription').val(); // Get transcription
  const context = $('#context').text(); // Get context
  const userPrompt = `${transcription}\n#Context:${context}`; // Combine both

  // Send transcription to Ollama server for EAD/XML generation
  const response = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'hf.co/Geraldine/FineLlama-3.2-3B-Instruct-ead-GGUF:Q5_K_M',
      prompt: `Generate EAD/XML for the following archival description: ${userPrompt}`,
      stream: false
    })
  });

  const data = await response.json();
  
  // Check if context is empty
  if (context.trim() === '') {
    // If context is empty, set the entire editor value
    editor.setValue(data.response);
  } else {
    // Get the current selection range in the editor
    const selectionRange = editor.getSelectionRange(); // Get the selection range

    // Replace only the highlighted text with the model's response
    editor.session.replace(selectionRange, data.response); 
  }
  
  $('#status').text('Ready');
  $('#loadingSpinner').hide(); // Hide the loading spinner
}

// Add this function to handle right-click context menu
function showContextMenu(event) {
  event.preventDefault(); // Prevent the default context menu

  const selectedText = editor.getSelectedText();
  if (selectedText) {
    // Create a context menu if it doesn't exist
    let $contextMenu = $('#contextMenu');
    if ($contextMenu.length === 0) {
      $contextMenu = $('<div>', {
        id: 'contextMenu',
        css: {
          position: 'absolute',
          backgroundColor: 'white',
          border: '1px solid #ccc',
          zIndex: 1000,
          display: 'none' // Initially hidden
        }
      }).append('<button id="addToContext">Add to Context</button>').appendTo('body');
    }

    // Position the context menu
    $contextMenu.css({
      left: `${event.pageX}px`,
      top: `${event.pageY}px`,
      display: 'block' // Show the context menu
    });

    // Add event listener for the "Add to Context" button
    $('#addToContext').off('click').on('click', () => {
      addToContext(selectedText);
      $contextMenu.hide(); // Hide the context menu
    });
  }
}

// Function to add selected text to the context div
function addToContext(selectedText) {
  $('#context').text(selectedText); // Populate the context div with the selected text
}

function formatXmlInEditor() {
  const xmlContent = editor.getValue();
  const formattedXml = formatXml(xmlContent);
  editor.setValue(formattedXml, 1);
}

function formatXml(xml, tab) {
  let formatted = '';
  let indentLevel = 0;
  tab = tab || '  '; // Use two spaces for indentation

  // Remove unnecessary spaces between tags and normalize newlines
  xml = xml.replace(/>\s*</g, '><').trim();

  // Split by tags
  xml.split(/(<[^>]+>)/g).forEach(node => {
    if (node.trim()) {
      if (node.startsWith('</')) {
        // Closing tag - decrease indentation
        indentLevel--;
        formatted += `${tab.repeat(indentLevel)}${node.trim()}\n`;
      } else if (node.startsWith('<') && !node.endsWith('/>')) {
        // Opening tag - add the tag and then increase indentation
        formatted += `${tab.repeat(indentLevel)}${node.trim()}\n`;
        indentLevel++;
      } else if (node.startsWith('<') && node.endsWith('/>')) {
        // Self-closing tag - add it at the current level
        formatted += `${tab.repeat(indentLevel)}${node.trim()}\n`;
      } else {
        // Text content - keep it at the current indentation level
        formatted += `${tab.repeat(indentLevel)}${node.trim()}\n`;
      }
    }
  });

  return formatted.trim();
}



// Event listeners
$('#startRecording').on('click', startRecording);
$('#stopRecording').on('click', stopRecording);

// Hide context menu on click elsewhere
$(document).on('click', () => {
  $('#contextMenu').hide();
});

// Add event listener for the send prompt button
$('#sendPrompt').on('click', sendPrompt);

// Add event listener for the prettify button
$('#prettifyXML').on('click', formatXmlInEditor);

// Initialize Whisper on page load
initWhisper();