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
    
    val boundsHistory: MutableList<List<Point>> = mutableListOf()
    var thrashScore by mutableDoubleStateOf(0.0)

    /**
     * Calculates the center of the card's current bounds.
     */
    fun getCenter(): Point {
        var sumX = 0.0
        var sumY = 0.0
        bounds.forEach { sumX += it.x; sumY += it.y }
        return Point(sumX / bounds.size, sumY / bounds.size)
    }

    /**
     * Updates the thrash score based on the historical stability of corners.
     */
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
                totalDist += calculateCornerError(prev, curr)
                pairs++
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
}

/**
 * Calculates the minimum distance between corners of two quads, 
 * accounting for the 4 possible rotations.
 */
fun calculateCornerError(q1: List<Point>, q2: List<Point>): Double {
    if (q1.size != 4 || q2.size != 4) return Double.MAX_VALUE
    
    val s1 = sortPointsClockwise(q1)
    val s2 = sortPointsClockwise(q2)
    
    var minTotalDist = Double.MAX_VALUE
    
    for (offset in 0 until 4) {
        var currentTotalDist = 0.0
        for (i in 0 until 4) {
            val p1 = s1[i]
            val p2 = s2[(i + offset) % 4]
            currentTotalDist += Math.sqrt(Math.pow(p1.x - p2.x, 2.0) + Math.pow(p1.y - p2.y, 2.0))
        }
        if (currentTotalDist < minTotalDist) {
            minTotalDist = currentTotalDist
        }
    }
    
    return minTotalDist / 4.0 // Return average corner error
}

/**
 * Rotates a sorted target quad to match the orientation of the current smoothed quad.
 */
fun matchRotation(sortedTarget: List<Point>, current: List<Point>): List<Point> {
    var bestOffset = 0
    var minTotalDist = Double.MAX_VALUE
    
    for (offset in 0 until 4) {
        var currentTotalDist = 0.0
        for (i in 0 until 4) {
            val p1 = current[i]
            val p2 = sortedTarget[(i + offset) % 4]
            currentTotalDist += Math.sqrt(Math.pow(p1.x - p2.x, 2.0) + Math.pow(p1.y - p2.y, 2.0))
        }
        if (currentTotalDist < minTotalDist) {
            minTotalDist = currentTotalDist
            bestOffset = offset
        }
    }
    
    return List(4) { i -> sortedTarget[(i + bestOffset) % 4] }
}

/**
 * Sorts four points in clockwise order starting from a consistent reference.
 */
fun sortPointsClockwise(points: List<Point>): List<Point> {
    if (points.size < 3) return points
    val sumX = points.sumOf { it.x }
    val sumY = points.sumOf { it.y }
    val centerX = sumX / points.size
    val centerY = sumY / points.size
    return points.sortedBy { Math.atan2(it.y - centerY, it.x - centerX) }
}

/**
 * Manages the persistent identity of cards across a temporal sequence of frames.
 */
class CardTracker {
    val activeCards = mutableListOf<TrackedCard>()
    private val CORNER_THRESHOLD_PX = 60.0 // Average distance per corner
    private val EXPIRY_MS = 600L 
    private val MIN_FRAMES_FOR_ID = 3 
    private val MAX_HISTORY = 5

    /**
     * Updates tracking state based on new geometric detections.
     * Matches new quads to existing tracks using minimum corner error.
     */
    fun updateGeometric(detectedQuads: List<List<Point>>): List<TrackedCard> {
        val now = System.currentTimeMillis()
        val unmatchedQuads = detectedQuads.toMutableList()
        
        // 1. Update existing tracks based on corner error (more robust than center distance)
        activeCards.forEach { tracked ->
            val bestMatch = unmatchedQuads.minByOrNull { quad -> 
                calculateCornerError(quad, tracked.bounds)
            }
            
            if (bestMatch != null && calculateCornerError(bestMatch, tracked.bounds) < CORNER_THRESHOLD_PX) {
                tracked.lastSeenTimestamp = now
                tracked.framesSeen++
                
                tracked.boundsHistory.add(bestMatch.map { it.clone() })
                if (tracked.boundsHistory.size > MAX_HISTORY) tracked.boundsHistory.removeAt(0)
                tracked.updateThrashScore()

                tracked.bounds = bestMatch.map { it.clone() }
                unmatchedQuads.remove(bestMatch)
            }
        }

        // 2. Remove expired tracks
        activeCards.removeAll { now - it.lastSeenTimestamp > EXPIRY_MS }

        // 3. Add new detections
        unmatchedQuads.forEach { bounds ->
            val newCard = TrackedCard(bounds = bounds)
            newCard.boundsHistory.add(bounds.map { it.clone() })
            activeCards.add(newCard)
        }

        return activeCards.toList()
    }

    /**
     * Identifies which tracked cards are eligible for ML identification.
     */
    fun getCardsForIdentification(): List<TrackedCard> {
        val now = System.currentTimeMillis()
        return activeCards.filter { 
            it.framesSeen >= MIN_FRAMES_FOR_ID && 
            (it.card == null || now - it.lastIdentifiedTimestamp > 1000L)
        }
    }

    /**
     * Assigns a physical SetCard identity to a temporal track.
     */
    fun updateIdentification(id: String, identifiedCard: SetCard?) {
        activeCards.find { it.id == id }?.apply {
            card = identifiedCard
            lastIdentifiedTimestamp = System.currentTimeMillis()
        }
    }
}
