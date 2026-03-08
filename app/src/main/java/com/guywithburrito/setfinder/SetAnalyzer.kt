package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.snapshots.SnapshotStateList
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.guywithburrito.setfinder.tracking.CardTracker
import com.guywithburrito.setfinder.tracking.TrackedCard
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

class SetAnalyzer(
    private val context: android.content.Context,
    private val scope: CoroutineScope,
    val settingsManager: com.guywithburrito.setfinder.tracking.SettingsManager
) : ImageAnalysis.Analyzer {
    private val historyPersistence = com.guywithburrito.setfinder.tracking.HistoryPersistence(context)
    private val seenSetsInSession = mutableSetOf<Set<com.guywithburrito.setfinder.card.SetCard>>()

    private val _detectedRects = mutableStateListOf<TrackedCard>()
    val detectedRects: SnapshotStateList<TrackedCard> = _detectedRects

    private val _foundSets = mutableStateListOf<List<TrackedCard>>()
    val foundSets: SnapshotStateList<List<TrackedCard>> = _foundSets
    
    private val _allCandidates = mutableStateListOf<List<Point>>()
    val allCandidates: SnapshotStateList<List<Point>> = _allCandidates

    var analysisWidth by mutableStateOf(1)
        private set
    var analysisHeight by mutableStateOf(1)
        private set
        
    var debugMode by mutableStateOf(false)
    var singleCardMode by mutableStateOf(false)
    var showLabels by mutableStateOf(settingsManager.showLabels)

    private val finder = CardFinder(settingsManager)
    private val unwarper = CardUnwarper()
    private val identifier = TFLiteCardIdentifier(context)
    private val tracker = CardTracker()

    // Decoupled ID state
    private val isIdentifying = AtomicBoolean(false)
    private var lastFrameForId: Mat? = null
    private var lastScaleForId: Double = 1.0

    override fun analyze(image: ImageProxy) {
        val frame: Mat = image.toMat()
        val rotationDegrees = image.imageInfo.rotationDegrees
        android.util.Log.d("SetAnalyzer", "ImageProxy: ${image.width}x${image.height}, rotation: $rotationDegrees")
        val rotatedFrame = Mat()
        if (rotationDegrees == 90) {
            Core.rotate(frame, rotatedFrame, Core.ROTATE_90_CLOCKWISE)
        } else if (rotationDegrees == 180) {
            Core.rotate(frame, rotatedFrame, Core.ROTATE_180)
        } else if (rotationDegrees == 270) {
            Core.rotate(frame, rotatedFrame, Core.ROTATE_90_COUNTERCLOCKWISE)
        } else {
            frame.copyTo(rotatedFrame)
        }
        
        analyzeMat(rotatedFrame)
        frame.release()
        rotatedFrame.release()
        image.close()
    }

    fun analyzeMat(frame: Mat) {
        val frameWidth = frame.cols()
        val frameHeight = frame.rows()
        
        val maxDim = 1000.0
        val scale = maxDim / Math.max(frameWidth.toDouble(), frameHeight.toDouble())
        val analysisFrame = Mat()
        if (scale < 1.0) {
            Imgproc.resize(frame, analysisFrame, Size(), scale, scale, Imgproc.INTER_AREA)
        } else {
            frame.copyTo(analysisFrame)
        }
        
        val actualW = analysisFrame.cols()
        val actualH = analysisFrame.rows()
        android.util.Log.d("SetAnalyzer", "Frame: ${frameWidth}x${frameHeight} -> Analysis: ${actualW}x${actualH}, scale=${scale}")

        scope.launch(Dispatchers.Main) {
            analysisWidth = actualW
            analysisHeight = actualH
        }

        // 1. FAST DETECTION AND TRACKING
        var quads = finder.findLikelyCards(analysisFrame)
        
        if (singleCardMode) {
            val center = Point(actualW / 2.0, actualH / 2.0)
            val best = quads.minByOrNull { q ->
                val p = q.toList()
                val qc = Point(p.sumOf { it.x } / 4.0, p.sumOf { it.y } / 4.0)
                Math.sqrt(Math.pow(qc.x - center.x, 2.0) + Math.pow(qc.y - center.y, 2.0))
            }
            quads = if (best != null) listOf(best) else emptyList()
        }

        val candidatePoints = quads.map { it.toList() }
        val trackedCards = tracker.updateGeometric(candidatePoints)

        // 2. TRIGGER OUT-OF-BAND IDENTIFICATION
        if (!isIdentifying.get()) {
            val cardsToId = tracker.getCardsForIdentification()
            if (cardsToId.isNotEmpty()) {
                lastFrameForId?.release()
                lastFrameForId = frame.clone() // frame is the high-res rotated frame
                lastScaleForId = scale
                startAsyncIdentification(cardsToId)
            }
        }
        
        // 3. FIND SETS among identified cards
        val identifiedOnly = trackedCards.filter { it.card != null }
        val threshold = 1.0f - settingsManager.sensitivity
        val sets = mutableListOf<List<TrackedCard>>()
        if (identifiedOnly.size >= 3) {
            for (i in 0 until identifiedOnly.size) {
                for (j in i + 1 until identifiedOnly.size) {
                    for (k in j + 1 until identifiedOnly.size) {
                        val c1 = identifiedOnly[i]; val c2 = identifiedOnly[j]; val c3 = identifiedOnly[k]
                        val card1 = c1.card!!; val card2 = c2.card!!; val card3 = c3.card!!
                        if (com.guywithburrito.setfinder.card.SetCard.isSet(card1, card2, card3)) {
                            val set = listOf(c1, c2, c3)
                            sets.add(set)
                            
                            // Session History: Save if this is a new set of attributes
                            val attributeSet = setOf(card1, card2, card3)
                            if (attributeSet !in seenSetsInSession) {
                                seenSetsInSession.add(attributeSet)
                                // Clone frame for capture
                                captureAndSaveSet(set, frame.clone())
                            }
                        }
                    }
                }
            }
        }
        analysisFrame.release()

        // 4. Update UI state on Main Thread
        scope.launch(Dispatchers.Main) {
            _detectedRects.clear()
            trackedCards.forEach { card ->
                // Keep original points; SetFinderView will scale them to the canvas
                _detectedRects.add(card.copy())
            }

            _foundSets.clear()
            sets.forEach { set ->
                _foundSets.add(set.map { it.copy() })
            }
            
            _allCandidates.clear()
            if (debugMode) {
                candidatePoints.forEach { quad ->
                    _allCandidates.add(quad.map { Point(it.x, it.y) })
                }
            }
        }
    }

    private fun captureAndSaveSet(set: List<TrackedCard>, frame: Mat) {
        scope.launch(Dispatchers.Default) {
            try {
                val cardPairs = set.mapNotNull { tracked ->
                    val card = tracked.card ?: return@mapNotNull null
                    val matOfPoint2f = MatOfPoint2f(*tracked.bounds.toTypedArray())
                    val warped = unwarper.unwarp(frame, matOfPoint2f)
                    val bmp = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
                    org.opencv.android.Utils.matToBitmap(warped, bmp)
                    warped.release()
                    card to bmp
                }
                if (cardPairs.size == 3) {
                    historyPersistence.saveSet(cardPairs)
                    Log.d("SetAnalyzer", "Saved new unique SET to history.")
                }
            } finally {
                frame.release()
            }
        }
    }

    private fun startAsyncIdentification(cards: List<TrackedCard>) {
        if (!isIdentifying.compareAndSet(false, true)) return
        
        scope.launch(Dispatchers.Default) {
            try {
                val frame = lastFrameForId ?: return@launch
                cards.forEach { tracked ->
                    // Quad is in analysisFrame (scaled) coordinates
                    val matOfPoint2f = MatOfPoint2f(*tracked.bounds.toTypedArray())
                    val warped = unwarper.unwarp(frame, matOfPoint2f)
                    
                    val bmp = Bitmap.createBitmap(warped.cols(), warped.rows(), Bitmap.Config.ARGB_8888)
                    org.opencv.android.Utils.matToBitmap(warped, bmp)
                    
                    val id = identifier.identifyCard(bmp)
                    if (id != null) {
                        tracker.updateIdentification(tracked.id, id)
                    }
                    
                    // Debug: Save chip to file (throttle to avoid perf issues)
                    if (System.currentTimeMillis() % 20 == 0L) {
                        val debugFile = java.io.File(context.cacheDir, "debug_chip.jpg")
                        try {
                            java.io.FileOutputStream(debugFile).use { out ->
                                bmp.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                            }
                            Log.d("SetAnalyzer", "Saved debug chip to: ${debugFile.absolutePath}")
                        } catch (e: Exception) {
                            Log.e("SetAnalyzer", "Failed to save debug chip", e)
                        }
                    }
                    warped.release()
                }
            } finally {
                isIdentifying.set(false)
            }
        }
    }

    private fun ImageProxy.toMat(): Mat {
        val bitmap = toBitmap()
        val mat = Mat()
        org.opencv.android.Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
        return mat
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
