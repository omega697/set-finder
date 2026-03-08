package com.guywithburrito.setfinder.tracking

import com.guywithburrito.setfinder.card.SetCard
import org.opencv.core.Point
import java.util.UUID

/**
 * Represents a card that is being tracked across frames.
 * Identification (the card field) can be updated out-of-band.
 */
data class TrackedCard(
    val id: String = UUID.randomUUID().toString(),
    var card: SetCard? = null,
    var bounds: List<Point>,
    var lastSeenTimestamp: Long = System.currentTimeMillis(),
    var framesSeen: Int = 1,
    var lastIdentifiedTimestamp: Long = 0
) {
    fun getCenter(): Point {
        var sumX = 0.0
        var sumY = 0.0
        bounds.forEach { sumX += it.x; sumY += it.y }
        return Point(sumX / bounds.size, sumY / bounds.size)
    }
}

class CardTracker {
    private val activeCards = mutableListOf<TrackedCard>()
    private val TRACKING_THRESHOLD_PX = 80.0 // Relaxed threshold for faster movement
    private val EXPIRY_MS = 600L 
    private val MIN_FRAMES_FOR_UI = 1 // Show quads immediately
    private val MIN_FRAMES_FOR_ID = 3 // Wait a bit before starting expensive ID

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
                // Fast smoothing
                tracked.bounds = tracked.bounds.zip(match).map { (old, new) ->
                    Point(old.x * 0.5 + new.x * 0.5, old.y * 0.5 + new.y * 0.5)
                }
                unmatchedQuads.remove(match)
            }
        }

        // 2. Remove expired tracks
        activeCards.removeAll { now - it.lastSeenTimestamp > EXPIRY_MS }

        // 3. Add new detections as potential tracks
        unmatchedQuads.forEach { bounds ->
            activeCards.add(TrackedCard(bounds = bounds))
        }

        return activeCards.toList()
    }

    /**
     * Finds tracked cards that need identification (unidentified or stale).
     */
    fun getCardsForIdentification(): List<TrackedCard> {
        return activeCards.filter { 
            it.framesSeen >= MIN_FRAMES_FOR_ID && 
            (it.card == null || System.currentTimeMillis() - it.lastIdentifiedTimestamp > 2000L)
        }
    }

    /**
     * Update a specific track with identified card data.
     */
    fun updateIdentification(id: String, identifiedCard: SetCard) {
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
