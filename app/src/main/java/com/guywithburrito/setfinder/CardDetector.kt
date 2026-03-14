package com.guywithburrito.setfinder

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.cv.OpenCVFrameProcessor
import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.tracking.CardTracker
import com.guywithburrito.setfinder.tracking.TrackedCard
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Stateful Vision Engine for detecting, tracking, and identifying Set cards.
 * Owns the temporal state (CardTracker) and orchestrates asynchronous ML 
 * identification while maintaining high-performance geometric tracking.
 */
class CardDetector(
    private val context: Context,
    private val scope: CoroutineScope,
    private val finder: QuadFinder,
    private val extractor: ChipExtractor = ChipExtractor(),
    private val identifier: CardIdentifier = CardIdentifier.getInstance(context),
    private val frameProcessor: FrameProcessor = OpenCVFrameProcessor(),
    private val tracker: CardTracker = CardTracker()
) {
    private val historyPersistence = com.guywithburrito.setfinder.tracking.HistoryPersistence(context)
    private val savedSetFingerprints = mutableSetOf<String>().apply {
        addAll(historyPersistence.getExistingFingerprints())
    }

    // Observable UI State
    private val _trackedCards = mutableStateListOf<TrackedCard>()
    val trackedCards: SnapshotStateList<TrackedCard> = _trackedCards

    private val _foundSets = mutableStateListOf<List<TrackedCard>>()
    val foundSets: SnapshotStateList<List<TrackedCard>> = _foundSets

    var analysisWidth by mutableStateOf(1000f)
        private set
    var analysisHeight by mutableStateOf(1000f)
        private set

    // Reactive Identification Pipeline
    private val idSignal = Channel<Unit>(Channel.CONFLATED)
    private var pendingFrame: Mat? = null
    private var pendingScale: Double = 1.0

    init {
        // 1. Continuous smoothing of boundaries in the background
        scope.launch {
            while (isActive) {
                tracker.activeCards.toList().forEach { it.updateSmoothing(it.bounds) }
                delay(16)
            }
        }

        // 2. Identification Worker (Reacts to fresh frames)
        scope.launch(Dispatchers.Default) {
            for (signal in idSignal) {
                val (frame, scale) = synchronized(this@CardDetector) {
                    val f = pendingFrame
                    pendingFrame = null // Take ownership and clear the buffer
                    f to pendingScale
                }
                
                if (frame == null) continue

                try {
                    val cardsToId = tracker.getCardsForIdentification()
                    cardsToId.forEach { tracked ->
                        val identified = identifySingleQuad(frame, tracked.bounds, scale)
                        tracker.updateIdentification(tracked.id, identified)
                    }
                } finally {
                    frame.release() // Ensure the cloned Mat is always released
                }
            }
        }
    }

    /**
     * Processes a single frame, updates tracking state, and signals for identification.
     * 
     * @param frame The raw camera frame to process.
     * @param singleCardMode If true, only the card closest to the center will be tracked.
     */
    fun processFrame(frame: Mat, singleCardMode: Boolean = false) {
        val maxDim = 1000.0
        val scale = maxDim / Math.max(frame.cols().toDouble(), frame.rows().toDouble())
        
        val analysisFrame = frameProcessor.createMat()
        frameProcessor.resize(frame, analysisFrame, Size(), scale, scale, Imgproc.INTER_AREA)

        analysisWidth = analysisFrame.cols().toFloat()
        analysisHeight = analysisFrame.rows().toFloat()
        
        // 1. Geometric Detection
        var quads = finder.findCandidates(analysisFrame)
        if (singleCardMode) {
            val center = Point(analysisFrame.cols() / 2.0, analysisFrame.rows() / 2.0)
            val best = quads.minByOrNull { q ->
                val p = q.toList()
                val qc = Point(p.sumOf { it.x } / 4.0, p.sumOf { it.y } / 4.0)
                sqrt((qc.x - center.x).pow(2.0) + (qc.y - center.y).pow(2.0))
            }
            quads = if (best != null) listOf(best) else emptyList()
        }

        // 2. Tracking Update
        val activeTracks = tracker.updateGeometric(quads.map { it.toList() })

        // 3. Buffer the latest frame and signal the identification worker
        if (tracker.getCardsForIdentification().isNotEmpty()) {
            synchronized(this) {
                pendingFrame?.release() // Close the previous one if it wasn't picked up yet
                pendingFrame = frame.clone() // Buffer the new one
                pendingScale = scale
            }
            idSignal.trySend(Unit)
        }

        // 4. Solve Game Logic
        val identifiedOnly = activeTracks.filter { it.card != null }
        val solvedSets = SetCard.findSets(identifiedOnly.map { it.card!! })
        val sets = solvedSets.map { setCards ->
            setCards.map { card -> identifiedOnly.first { it.card == card } }
        }

        // 5. Automatic History Capture
        sets.forEach { setTracks ->
            val fingerprint = setTracks.mapNotNull { it.card?.let { c -> "${c.shape}|${c.pattern}|${c.count}|${c.color}" } }
                .sorted()
                .joinToString("||")
            
            if (fingerprint.isNotEmpty() && !savedSetFingerprints.contains(fingerprint)) {
                savedSetFingerprints.add(fingerprint)
                val frameClone = frame.clone() // Clone here to avoid race condition with release()
                scope.launch(Dispatchers.IO) {
                    try {
                        val cardsToSave = setTracks.mapNotNull { track ->
                            track.card?.let { card ->
                                card to extractChip(frameClone, track, scale)
                            }
                        }
                        if (cardsToSave.size == 3) {
                            historyPersistence.saveSet(cardsToSave)
                        }
                    } finally {
                        frameClone.release()
                    }
                }
            }
        }

        analysisFrame.release()

        // 6. Publish to UI
        scope.launch(Dispatchers.Main) {
            _trackedCards.clear(); _trackedCards.addAll(activeTracks)
            _foundSets.clear(); _foundSets.addAll(sets)
        }
    }

    /**
     * Synchronous identification (for history capture and testing).
     * 
     * @param frame Full-resolution frame.
     * @param quads List of quads to identify.
     * @param scale Scaling factor from analysis space to full-resolution.
     * @return List of identified cards (null if identification failed).
     */
    fun identifyQuadsSync(frame: Mat, quads: List<List<Point>>, scale: Double): List<SetCard?> {
        return quads.map { identifySingleQuad(frame, it, scale) }
    }

    /**
     * Extracts and identifies a single card from a frame.
     */
    private fun identifySingleQuad(frame: Mat, points: List<Point>, scale: Double): SetCard? {
        val corners = points.map { p -> Point(p.x / scale, p.y / scale) }
        val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
        val chip = extractor.extract(frame, fullResQuad)
        val result = identifier.identifyCard(chip)
        fullResQuad.release()
        return result
    }

    /**
     * Extracts a chip for persistence (history). 
     * Uses the internal scaling context to map tracked bounds to the provided frame.
     */
    fun extractChip(frame: Mat, tracked: TrackedCard, scale: Double): Bitmap {
        val corners = tracked.bounds.map { p -> Point(p.x / scale, p.y / scale) }
        val fullResQuad = frameProcessor.createMatOfPoint2f(corners)
        val chip = extractor.extract(frame, fullResQuad)
        fullResQuad.release()
        return chip
    }

    /**
     * Releases native resources.
     */
    fun close() {
        identifier.close()
        idSignal.close()
        synchronized(this) {
            pendingFrame?.release()
            pendingFrame = null
        }
    }
}
