package com.guywithburrito.setfinder.ui

import android.content.Context
import android.util.Log
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.guywithburrito.setfinder.SetAnalyzer
import com.guywithburrito.setfinder.tracking.TrackedCard
import org.opencv.core.Point
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.ui.text.font.FontWeight

@Composable
fun SetFinderView(
    onSettingsClicked: () -> Unit = {},
    onHistoryClicked: () -> Unit = {}
) {
    val coroutineScope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    val settingsManager = remember { com.guywithburrito.setfinder.tracking.SettingsManager(context) }
    val setAnalyzer = remember { SetAnalyzer(context, coroutineScope, settingsManager) }
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }

    var sensitivity by remember { mutableFloatStateOf(settingsManager.sensitivity) }
    var showLabels by remember { mutableStateOf(settingsManager.showLabels) }
    var showDebug by remember { mutableStateOf(false) }
    var singleCardMode by remember { mutableStateOf(false) }

    LaunchedEffect(showDebug, singleCardMode, showLabels) {
        setAnalyzer.debugMode = showDebug
        setAnalyzer.singleCardMode = singleCardMode
        setAnalyzer.showLabels = showLabels
    }

    val textPaint = remember {
        Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 40f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK)
        }
    }

    Column(modifier = Modifier.fillMaxSize().background(MaterialTheme.colors.background)) {
        // Top: Scanning Window
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        this.scaleType = PreviewView.ScaleType.FIT_CENTER
                        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    }
                },
                modifier = Modifier.fillMaxSize(),
                update = { previewView ->
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                            .build()
                            .also {
                                it.setAnalyzer(analyzerExecutor, setAnalyzer)
                            }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                CameraSelector.DEFAULT_BACK_CAMERA,
                                preview,
                                imageAnalysis
                            )
                        } catch (e: Exception) {
                            Log.e("SetFinderView", "Use case binding failed", e)
                        }
                    }, ContextCompat.getMainExecutor(context))
                }
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                val imgWidth = setAnalyzer.analysisWidth.toFloat()
                val imgHeight = setAnalyzer.analysisHeight.toFloat()
                if (imgWidth <= 1f || imgHeight <= 1f) return@Canvas

                val scale = Math.min(size.width / imgWidth, size.height / imgHeight)
                val offsetX = (size.width - (imgWidth * scale)) / 2f
                val offsetY = (size.height - (imgHeight * scale)) / 2f
                
                fun Point.toOffset() = Offset((x.toFloat() * scale) + offsetX, (y.toFloat() * scale) + offsetY)

                // Debug Candidates
                if (showDebug) {
                    setAnalyzer.allCandidates.forEach { points ->
                        for (i in 0 until points.size) {
                            drawLine(
                                color = Color.Red.copy(alpha = 0.3f),
                                start = points[i].toOffset(),
                                end = points[(i + 1) % points.size].toOffset(),
                                strokeWidth = 1f
                            )
                        }
                    }
                }

                // Tracked Cards
                setAnalyzer.detectedRects.forEach { card ->
                    if (showDebug || card.card != null) {
                        val points = card.bounds
                        for (i in 0 until points.size) {
                            drawLine(
                                color = if (showDebug) Color.Yellow else Color.White.copy(alpha = 0.5f),
                                start = points[i].toOffset(),
                                end = points[(i + 1) % points.size].toOffset(),
                                strokeWidth = if (showDebug) 4f else 2f
                            )
                        }

                        if (showLabels) {
                            val center = card.getCenter().toOffset()
                            card.card?.let { identified ->
                                drawContext.canvas.nativeCanvas.drawText("${identified.count} ${identified.color}", center.x - 60f, center.y, textPaint)
                                drawContext.canvas.nativeCanvas.drawText("${identified.pattern} ${identified.shape}", center.x - 60f, center.y + 40f, textPaint)
                            }
                        }
                    }
                }

                // Sets
                setAnalyzer.foundSets.forEachIndexed { idx, set ->
                    val colors = settingsManager.highlightColors.map { it.second }
                    val color = if (colors.isNotEmpty()) colors[idx % colors.size] else Color.Green
                    set.forEach { card ->
                        for (i in 0 until card.bounds.size) {
                            drawLine(color = color, start = card.bounds[i].toOffset(), end = card.bounds[(i+1)%4].toOffset(), strokeWidth = 8f)
                        }
                    }
                }
                
                if (singleCardMode) {
                    val cx = size.width / 2f; val cy = size.height / 2f
                    val w = 400f * (size.width / 1080f); val h = 600f * (size.height / 2424f)
                    drawRect(color = Color.Yellow, topLeft = Offset(cx - w/2, cy - h/2), size = androidx.compose.ui.geometry.Size(w, h), style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4f))
                }
            }
        }

        // Bottom: Controls
        Surface(
            elevation = 8.dp,
            color = MaterialTheme.colors.surface,
            contentColor = MaterialTheme.colors.onSurface
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    ControlToggle(
                        icon = Icons.Default.BugReport,
                        isActive = showDebug,
                        onClick = { showDebug = !showDebug },
                        label = "Debug"
                    )
                    ControlToggle(
                        icon = Icons.Default.CropPortrait,
                        isActive = singleCardMode,
                        onClick = { singleCardMode = !singleCardMode },
                        label = "Single"
                    )
                    ControlToggle(
                        icon = Icons.Default.Label,
                        isActive = showLabels,
                        onClick = { 
                            showLabels = !showLabels
                            settingsManager.showLabels = showLabels
                        },
                        label = "Labels"
                    )
                    IconButton(onClick = onHistoryClicked) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(Icons.Default.History, contentDescription = "History")
                            Text("History", style = MaterialTheme.typography.caption)
                        }
                    }
                    IconButton(onClick = onSettingsClicked) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                            Text("Settings", style = MaterialTheme.typography.caption)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
                
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Sensitivity", style = MaterialTheme.typography.body2, modifier = Modifier.width(80.dp))
                    Slider(
                        value = sensitivity,
                        onValueChange = { 
                            sensitivity = it
                            settingsManager.sensitivity = it
                        },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
fun ControlToggle(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    isActive: Boolean,
    onClick: () -> Unit,
    label: String
) {
    IconButton(onClick = onClick) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                icon,
                contentDescription = label,
                tint = if (isActive) MaterialTheme.colors.primary else MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
            )
            Text(
                label,
                style = MaterialTheme.typography.caption,
                color = if (isActive) MaterialTheme.colors.primary else MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
            )
        }
    }
}
