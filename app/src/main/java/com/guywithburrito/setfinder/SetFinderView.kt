package com.guywithburrito.setfinder

import android.content.Context
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.FloatingActionButton
import androidx.compose.material.Icon
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

@Composable
@androidx.compose.ui.tooling.preview.Preview
fun SetFinderView (modifier: Modifier = Modifier,
                   scaleType: PreviewView.ScaleType = PreviewView.ScaleType.FILL_CENTER
) {
    val coroutineScope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    val setAnalyzer = SetAnalyzer()
    Box(modifier = Modifier) {
        AndroidView(
            modifier = modifier,
            factory = { context ->
                val previewView = PreviewView(context).apply {
                    this.scaleType = scaleType
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }

                // CameraX Preview UseCase
                val previewUseCase = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val analysisUseCase = ImageAnalysis.Builder().build().also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor(), setAnalyzer)
                }

                coroutineScope.launch {
                    val cameraProvider = context.getCameraProvider()
                    try {
                        // Must unbind the use-cases before rebinding them.
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner, cameraSelector, previewUseCase, analysisUseCase
                        )
                    } catch (ex: Exception) {
                        Log.e("CameraPreview", "Use case binding failed", ex)
                    }
                }

                previewView
            }
        )
        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            // Draw the image here?
            drawLine(
                start = Offset(x = canvasWidth, y = 0f),
                end = Offset(x = 0f, y = canvasHeight),
                color = Color.Blue
            )
        }
        FloatingActionButton(
            modifier = Modifier
                .wrapContentSize()
                .padding(16.dp)
                .align(Alignment.BottomEnd),
            onClick = {
                Log.d("MainActivity", "TODO: Change camera.")
            }
        ) {
            Icon(
                imageVector = Icons.Default.Cameraswitch,
                contentDescription = "Switch cameras",
                modifier = Modifier,
            )
        }
    }
}

suspend fun Context.getCameraProvider(): ProcessCameraProvider = suspendCoroutine { continuation ->
    ProcessCameraProvider.getInstance(this).also { future ->
        future.addListener({
            continuation.resume(future.get())
        }, executor)
    }
}

val Context.executor: Executor
    get() = ContextCompat.getMainExecutor(this)