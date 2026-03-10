package com.guywithburrito.setfinder

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.snapshots.SnapshotStateList
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.cv.OpenCVQuadFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.tracking.CardTracker
import com.guywithburrito.setfinder.tracking.TrackedCard
import com.guywithburrito.setfinder.tracking.HistoryPersistence
import com.guywithburrito.setfinder.tracking.SettingsManager
import com.guywithburrito.setfinder.card.SetCard
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.*
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Orchestrates the real-time Set detection pipeline across multiple camera frames.
 * Manages temporal tracking, asynchronous identification, and game-rule solving.
 * High-performance design: Synchronous fast-path for geometric tracking, 
 * Asynchronous slow-path for trait identification.
 */
class SetAnalyzer(
    private val context: Context,
    private val scope: CoroutineScope,
    val settingsManager: SettingsManager,
    private val tracker: CardTracker = CardTracker(),
    private val frameProcessor: FrameProcessor = OpenCVFrameProcessor(),
    private val finder: QuadFinder = OpenCVQuadFinder(settingsManager),
    private val extractor: ChipExtractor = ChipExtractor(),
    private val identifier: CardIdentifier = CardIdentifier.getInstance(context),
    private val detector: SetDetector = SetDetector(finder, extractor, identifier, frameProcessor)
) : ImageAnalysis.Analyzer {
    private val historyPersistence = HistoryPersistence(context)
    private val seenSetsInSession = mutableSetOf<Set<SetCard>>()

    private val _detectedRects = mutableStateListOf<TrackedCard>()
    val detectedRects: SnapshotStateList<TrackedCard> = _detectedRects

    private val _foundSets = mutableStateListOf<List<TrackedCard>>()
    val foundSets: SnapshotStateList<List<TrackedCard>> = _foundSets
    
    private val _allCandidates = mutableStateListOf<List<Point>>()
    val allCandidates: SnapshotStateList<List<Point>> = _allCandidates

    var analysisWidth by mutableStateOf(1000f)
        private set
    var analysisHeight by mutableStateOf(1000f)
        private set
        
    var debugMode by mutableStateOf(false)
    var singleCardMode by mutableStateOf(false)
    var showLabels by mutableStateOf(settingsManager.showLabels)

    private val isIdentifying = AtomicBoolean(false)
    private var lastFrameForId: Mat? = null
    private var lastScaleForId: Double = 1.0

    init {
        scope.launch {
            while (isActive) {
                // Background smoothing of tracked card boundaries
                tracker.activeCards.toList().forEach { it.updateSmoothing(it.bounds) }
                kotlinx.coroutines.delay(16)
            }
        }
    }

    fun analyzeARFrame(frame: com.google.ar.core.Frame) {
        if (isIdentifying.get()) return
        
        scope.launch(Dispatchers.Default) {
            try {
                val image = frame.acquireCameraImage() ?: return@launch
                val nv21 = nv21(image.planes[0].buffer, image.planes[1].buffer, image.planes[2].buffer)
                val yuvMat = frameProcessor.createMat()
                yuvMat.put(0, 0, nv21)
                
                val frameMat = frameProcessor.createMat()
                frameProcessor.yuvToRgb(yuvMat, frameMat)
                
                val portraitMat = frameProcessor.createMat()
                frameProcessor.rotate(frameMat, portraitMat, FrameProcessor.ROTATE_90_CLOCKWISE)
                
                analyzeMat(portraitMat)
                
                portraitMat.release(); frameMat.release(); yuvMat.release()
                image.close()
            } catch (e: Exception) {
                Log.e("SetAnalyzer", "AR Analysis failed", e)
            }
        }
    }

    fun analyzeMat(frame: Mat) {
        val maxDim = 1000.0
        val scale = maxDim / Math.max(frame.cols().toDouble(), frame.rows().toDouble())
        val analysisFrame = frameProcessor.createMat()
        frameProcessor.resize(frame, analysisFrame, Size(), scale, scale, Imgproc.INTER_AREA)

        analysisWidth = analysisFrame.cols().toFloat()
        analysisHeight = analysisFrame.rows().toFloat()
        
        // 1. FAST PATH: Geometric Detection (Every Frame)
        var quads = detector.detectQuads(analysisFrame)
        
        if (singleCardMode) {
            val center = Point(analysisFrame.cols() / 2.0, analysisFrame.rows() / 2.0)
            val best = quads.minByOrNull { q ->
                val qc = Point(q.sumOf { it.x } / 4.0, q.sumOf { it.y } / 4.0)
                Math.sqrt(Math.pow(qc.x - center.x, 2.0) + Math.pow(qc.y - center.y, 2.0))
            }
            quads = if (best != null) listOf(best) else emptyList()
        }

        // 2. Temporal Tracking (Stable identities across frames)
        val trackedCards = tracker.updateGeometric(quads)

        // 3. SLOW PATH: Selective Asynchronous Identification
        if (!isIdentifying.get()) {
            val cardsToId = tracker.getCardsForIdentification()
            if (cardsToId.isNotEmpty()) {
                lastFrameForId?.release()
                lastFrameForId = frame.clone()
                lastScaleForId = scale
                startAsyncIdentification(cardsToId)
            }
        }
        
        // 4. Domain Logic: Game Rule Solving (on tracked stable cards)
        val identifiedOnly = trackedCards.filter { it.card != null }
        val identifiedCards = identifiedOnly.map { it.card!! }
        val solvedSets = SetCard.findSets(identifiedCards)
        
        val sets = mutableListOf<List<TrackedCard>>()
        solvedSets.forEach { setCards ->
            val trackedSet = setCards.map { card -> identifiedOnly.first { it.card == card } }
            sets.add(trackedSet)
            
            // Persist new sets found in this session
            if (setCards.toSet() !in seenSetsInSession) {
                seenSetsInSession.add(setCards.toSet())
                captureAndSaveSet(trackedSet, frame.clone(), scale)
            }
        }
        analysisFrame.release()

        // 5. Update UI state (Observed by Compose)
        scope.launch(Dispatchers.Main) {
            _detectedRects.clear(); _detectedRects.addAll(trackedCards)
            _foundSets.clear(); _foundSets.addAll(sets)
            _allCandidates.clear()
            if (debugMode) quads.forEach { _allCandidates.add(it) }
        }
    }

    private fun startAsyncIdentification(cards: List<TrackedCard>) {
        if (!isIdentifying.compareAndSet(false, true)) return
        
        scope.launch(Dispatchers.Default) {
            try {
                val fullFrame = lastFrameForId ?: return@launch
                val quadsToId = cards.map { it.bounds }
                
                // Identify traits using the detector's vision pipeline
                val results = detector.identifyQuads(fullFrame, quadsToId, lastScaleForId)
                
                results.forEachIndexed { index, result ->
                    tracker.updateIdentification(cards[index].id, result)
                }
            } finally {
                isIdentifying.set(false)
            }
        }
    }

    private fun captureAndSaveSet(set: List<TrackedCard>, frame: Mat, scale: Double) {
        scope.launch(Dispatchers.Default) {
            try {
                val quads = set.map { it.bounds }
                val results = detector.identifyQuads(frame, quads, scale)
                
                val cardsWithImages = set.mapIndexedNotNull { index, tracked ->
                    results[index]?.let { identified ->
                        val corners = tracked.bounds.map { p -> Point(p.x / scale, p.y / scale) }
                        val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
                        val chip = detector.extractChipForPersistence(frame, fullResQuad)
                        fullResQuad.release()
                        identified to chip
                    }
                }
                historyPersistence.saveSet(cardsWithImages)
            } finally {
                frame.release()
            }
        }
    }

    /**
     * Internal helper to bridge camera frames to OpenCV Mats.
     */
    override fun analyze(image: ImageProxy) {
        val frame: Mat = image.toMat()
        val rotationDegrees = image.imageInfo.rotationDegrees
        val rotatedFrame = frameProcessor.createMat()
        when (rotationDegrees) {
            90 -> frameProcessor.rotate(frame, rotatedFrame, FrameProcessor.ROTATE_90_CLOCKWISE)
            180 -> frameProcessor.rotate(frame, rotatedFrame, FrameProcessor.ROTATE_180)
            270 -> frameProcessor.rotate(frame, rotatedFrame, FrameProcessor.ROTATE_90_COUNTERCLOCKWISE)
            else -> frame.copyTo(rotatedFrame)
        }
        
        analyzeMat(rotatedFrame)
        frame.release(); rotatedFrame.release()
        image.close()
    }

    private fun ImageProxy.toMat(): Mat {
        val bitmap = this.toBitmap()
        val mat = frameProcessor.createMat()
        org.opencv.android.Utils.bitmapToMat(bitmap, mat)
        val rgb = frameProcessor.createMat()
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_RGBA2RGB)
        mat.release()
        return rgb
    }

    private fun ImageProxy.toBitmap(): Bitmap {
        val nv21 = nv21(planes[0].buffer, planes[1].buffer, planes[2].buffer)
        val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun nv21(y: ByteBuffer, u: ByteBuffer, v: ByteBuffer): ByteArray {
        val nv21 = ByteArray(y.remaining() + u.remaining() + v.remaining())
        y.get(nv21, 0, y.remaining())
        v.get(nv21, y.remaining(), v.remaining())
        u.get(nv21, y.remaining() + v.remaining(), u.remaining())
        return nv21
    }
}
