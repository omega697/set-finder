package com.guywithburrito.setfinder.ui

import android.util.Log
import android.view.ViewGroup
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.guywithburrito.setfinder.CardDetector
import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.tracking.SettingsManager
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.node.Node
import io.github.sceneview.math.Position
import dev.romainguy.kotlin.math.Quaternion
import com.google.ar.core.Plane
import io.github.sceneview.math.Scale
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

@Composable
fun ARSetFinderView(
    onSettingsClicked: () -> Unit = {},
    onHistoryClicked: () -> Unit = {}
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val settingsManager = remember { SettingsManager(context) }
    val cardDetector = remember { 
        CardDetector(
            context = context,
            scope = coroutineScope,
            finder = QuadFinder.getInstance(context)
        )
    }
    
    var sensitivity by remember { mutableFloatStateOf(settingsManager.sensitivity) }
    var showLabels by remember { mutableStateOf(settingsManager.showLabels) }
    var showDebug by remember { mutableStateOf(false) }

    val cardNodes = remember { mutableMapOf<String, Node>() }

    Column(modifier = Modifier.fillMaxSize().background(MaterialTheme.colors.background)) {
        Box(modifier = Modifier.weight(1f).fillMaxWidth()) {
            AndroidView(
                factory = { ctx ->
                    ARSceneView(ctx).apply {
                        layoutParams = ViewGroup.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        planeRenderer.isVisible = false
                    }
                },
                modifier = Modifier.fillMaxSize(),
                update = { view ->
                    view.onSessionUpdated = { _, frame ->
                        try {
                            // Extract frame for CV analysis
                            val image = frame.acquireCameraImage()
                            val mat = Mat(image.height + image.height / 2, image.width, org.opencv.core.CvType.CV_8UC1)
                            // This conversion is simplified; in a real app, you'd use a more robust YUV->Mat converter
                            // For now, we'll use a placeholder to maintain architectural flow.
                            // image.planes...
                            
                            cardDetector.processFrame(mat)
                            mat.release()
                            image.close()
                        } catch (e: Exception) {
                            // Log.e("ARSetFinder", "Frame capture failed", e)
                        }

                        val activeCards = cardDetector.trackedCards.toList()
                        val activeIds = activeCards.map { it.id }.toSet()
                        
                        val iterator = cardNodes.entries.iterator()
                        while (iterator.hasNext()) {
                            val entry = iterator.next()
                            if (!activeIds.contains(entry.key)) {
                                view.removeChildNode(entry.value)
                                iterator.remove()
                            }
                        }

                        activeCards.forEach { card ->
                            val center = card.getCenter()
                            val normX = center.x.toFloat() / cardDetector.analysisWidth
                            val normY = center.y.toFloat() / cardDetector.analysisHeight
                            
                            val hitResult = frame.hitTest(normX * view.width, normY * view.height)
                                .firstOrNull { hit ->
                                    val trackable = hit.trackable
                                    trackable is Plane && trackable.isPoseInPolygon(hit.hitPose)
                                }

                            if (hitResult != null) {
                                val pose = hitResult.hitPose
                                val node = cardNodes.getOrPut(card.id) {
                                    Node(view.engine).apply {
                                        view.addChildNode(this)
                                    }
                                }
                                
                                node.position = Position(pose.tx(), pose.ty(), pose.tz())
                                node.quaternion = Quaternion(pose.qx(), pose.qy(), pose.qz(), pose.qw())
                                node.scale = Scale(0.064f, 0.005f, 0.089f)
                            }
                        }
                    }
                }
            )

            if (showDebug) {
                Box(modifier = Modifier.padding(16.dp).background(Color.Black.copy(alpha = 0.5f))) {
                    Text("AR Mode (World Anchored)", color = Color.White, modifier = Modifier.padding(8.dp))
                }
            }
        }

        Surface(elevation = 8.dp, color = MaterialTheme.colors.surface) {
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
