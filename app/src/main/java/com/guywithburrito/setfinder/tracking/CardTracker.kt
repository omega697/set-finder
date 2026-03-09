package com.guywithburrito.setfinder.tracking

import com.guywithburrito.setfinder.card.SetCard
import org.opencv.core.Point
import java.util.UUID

import androidx.compose.runtime.*

/**
 * Represents a card that is being tracked across frames.
 * Identification (the card field) can be updated out-of-band.
 */
class TrackedCard(
    val id: String = UUID.randomUUID().toString(),
    card: SetCard? = null,
    bounds: List<Point>, // Raw detected bounds
    lastSeenTimestamp: Long = System.currentTimeMillis(),
    framesSeen: Int = 1,
    lastIdentifiedTimestamp: Long = 0
) {
    var card by mutableStateOf(card)
    var bounds by mutableStateOf(bounds)
    // ALWAYS initialize smoothedBounds as a sorted quad to prevent bowties from frame 1
    var smoothedBounds by mutableStateOf(sortPointsClockwise(bounds).map { it.clone() })
    var lastSeenTimestamp by mutableLongStateOf(lastSeenTimestamp)
    var framesSeen by mutableIntStateOf(framesSeen)
    var lastIdentifiedTimestamp by mutableLongStateOf(lastIdentifiedTimestamp)
    
    val velocities: List<Point> = listOf(Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0), Point(0.0, 0.0))
    val boundsHistory: MutableList<List<Point>> = mutableListOf()
    var thrashScore by mutableDoubleStateOf(0.0)

    fun getCenter(): Point {
        var sumX = 0.0
        var sumY = 0.0
        bounds.forEach { sumX += it.x; sumY += it.y }
        return Point(sumX / bounds.size, sumY / bounds.size)
    }

    fun updateThrashScore() {
        if (boundsHistory.size < 2) {
            thrashScore = 0.0
            return
        }
        
        var totalDist = 0.0
        var pairs = 0
        for (h in 1 until boundsHistory.size) {
            val prev = boundsHistory[h-1]
            val curr = boundsHistory[h]
            if (prev.size == curr.size) {
                for (p in 0 until prev.size) {
                    val dX = prev[p].x - curr[p].x
                    val dY = prev[p].y - curr[p].y
                    totalDist += Math.sqrt(dX * dX + dY * dY)
                    pairs++
                }
            }
        }
        thrashScore = if (pairs > 0) totalDist / pairs else 0.0
    }

    /**
     * Apply smoothing to smoothedBounds towards target bounds.
     * Uses a simple Lerp for stability, with asymmetric bias.
     */
    fun updateSmoothing(target: List<Point>) {
        if (target.size != 4) return
        
        val center = getCenter()
        val newSmoothed = mutableListOf<Point>()
        
        // Match the target's rotation to our current smoothed quad
        val matchedTarget = matchRotation(sortPointsClockwise(target), smoothedBounds)

        for (i in 0 until 4) {
            val s = smoothedBounds[i]
            val t = matchedTarget[i]
            
            // Vector from center to current and target
            val dS = Math.sqrt(Math.pow(s.x - center.x, 2.0) + Math.pow(s.y - center.y, 2.0))
            val dT = Math.sqrt(Math.pow(t.x - center.x, 2.0) + Math.pow(t.y - center.y, 2.0))
            
            // Asymmetric smoothing: fast expand, slow contract (snappier)
            val alpha = if (dT > dS) 0.6 else 0.4
            
            newSmoothed.add(Point(
                s.x + (t.x - s.x) * alpha,
                s.y + (t.y - s.y) * alpha
            ))
        }
        smoothedBounds = newSmoothed
    }

    private fun matchRotation(sortedTarget: List<Point>, current: List<Point>): List<Point> {
        var bestOffset = 0
        var minTotalDist = Double.MAX_VALUE
        
        for (offset in 0 until 4) {
            var currentTotalDist = 0.0
            for (i in 0 until 4) {
                currentTotalDist += dist(current[i], sortedTarget[(i + offset) % 4])
            }
            if (currentTotalDist < minTotalDist) {
                minTotalDist = currentTotalDist
                bestOffset = offset
            }
        }
        
        return List(4) { i -> sortedTarget[(i + bestOffset) % 4] }
    }
}

private fun sortPointsClockwise(points: List<Point>): List<Point> {
    if (points.size < 3) return points
    val sumX = points.sumOf { it.x }
    val sumY = points.sumOf { it.y }
    val centerX = sumX / points.size
    val centerY = sumY / points.size
    return points.sortedBy { Math.atan2(it.y - centerY, it.x - centerX) }
}

private fun dist(p1: Point, p2: Point): Double {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2.0) + Math.pow(p1.y - p2.y, 2.0))
}

private fun getCenter(quad: List<Point>): Point {
    var sumX = 0.0
    var sumY = 0.0
    quad.forEach { sumX += it.x; sumY += it.y }
    return Point(sumX / quad.size, sumY / quad.size)
}

class CardTracker {
    val activeCards = mutableListOf<TrackedCard>()
    private val TRACKING_THRESHOLD_PX = 80.0 // Relaxed threshold for faster movement
    private val EXPIRY_MS = 600L 
    private val MIN_FRAMES_FOR_UI = 1 // Show quads immediately
    private val MIN_FRAMES_FOR_ID = 3 // Wait a bit before starting expensive ID
    private val MAX_HISTORY = 5

    /**
     * Update tracking with ONLY geometric detections (quads).
     */
    fun updateGeometric(detectedQuads: List<List<Point>>): List<TrackedCard> {
        val now = System.currentTimeMillis()
        val unmatchedQuads = detectedQuads.toMutableList()
        
        // 1. Update existing tracks based on proximity
        activeCards.forEach { tracked ->
            val trackedCenter = tracked.getCenter()
            val match = unmatchedQuads.minByOrNull { quad -> 
                dist(getCenter(quad), trackedCenter)
            }
            
            if (match != null && dist(getCenter(match), trackedCenter) < TRACKING_THRESHOLD_PX) {
                tracked.lastSeenTimestamp = now
                tracked.framesSeen++
                
                // Track history BEFORE smoothing for accurate thrash measurement
                tracked.boundsHistory.add(match.map { it.clone() })
                if (tracked.boundsHistory.size > MAX_HISTORY) tracked.boundsHistory.removeAt(0)
                tracked.updateThrashScore()

                // High-quality smoothing with inertia - Target updated here, loop does the moving
                tracked.bounds = match.map { it.clone() }
                
                unmatchedQuads.remove(match)
            }
        }

        // 2. Remove expired tracks
        activeCards.removeAll { now - it.lastSeenTimestamp > EXPIRY_MS }

        // 3. Add new detections as potential tracks
        unmatchedQuads.forEach { bounds ->
            val newCard = TrackedCard(bounds = bounds)
            newCard.boundsHistory.add(bounds.map { it.clone() })
            activeCards.add(newCard)
        }

        return activeCards.toList()
    }

    /**
     * Finds tracked cards that need identification (unidentified or stale).
     */
    fun getCardsForIdentification(): List<TrackedCard> {
        val now = System.currentTimeMillis()
        return activeCards.filter { 
            it.framesSeen >= MIN_FRAMES_FOR_ID && 
            (it.card == null || now - it.lastIdentifiedTimestamp > 1000L)
        }
    }

    /**
     * Update a specific track with identified card data.
     * Passing null clears the identity (e.g. if the object is no longer seen as a card).
     */
    fun updateIdentification(id: String, identifiedCard: SetCard?) {
        activeCards.find { it.id == id }?.apply {
            card = identifiedCard
            lastIdentifiedTimestamp = System.currentTimeMillis()
        }
    }

    private fun getCenter(bounds: List<Point>): Point {
        var sumX = 0.0
        var sumY = 0.0
        bounds.forEach { sumX += it.x; sumY += it.y }
        return Point(sumX / bounds.size, sumY / bounds.size)
    }

    private fun dist(p1: Point, p2: Point): Double {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2.0) + Math.pow(p1.y - p2.y, 2.0))
    }
}
