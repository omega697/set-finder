package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.tracking.CardTracker
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.opencv.core.Point

class CardTrackerTest {

    private val boundsA = listOf(Point(0.0, 0.0), Point(10.0, 0.0), Point(10.0, 10.0), Point(0.0, 10.0))

    @Test
    fun tracker_returnsCardsImmediately() {
        val tracker = CardTracker()
        
        // 1st frame - Should return tracked cards immediately with framesSeen=1
        val tracked = tracker.updateGeometric(listOf(boundsA))
        assertThat(tracked).hasSize(1)
        assertThat(tracked[0].framesSeen).isEqualTo(1)
    }

    @Test
    fun tracker_matchesProximity() {
        val tracker = CardTracker()
        
        // Initial detection
        tracker.updateGeometric(listOf(boundsA))

        // Second detection slightly offset (well within 80px threshold)
        val boundsB = listOf(Point(5.0, 5.0), Point(15.0, 5.0), Point(15.0, 15.0), Point(5.0, 15.0))
        val tracked = tracker.updateGeometric(listOf(boundsB))

        assertThat(tracked).hasSize(1)
        assertThat(tracked[0].framesSeen).isEqualTo(2)
        // Raw bounds should match the detection
        assertThat(tracked[0].bounds[0].x).isEqualTo(5.0)
    }

    @Test
    fun tracker_smoothingIsAsymmetric() {
        val tracker = CardTracker()
        
        // Start with a small quad at center
        val small = listOf(
            Point(90.0, 90.0), Point(110.0, 90.0),
            Point(110.0, 110.0), Point(90.0, 110.0)
        )
        tracker.updateGeometric(listOf(small))
        val card = tracker.activeCards[0]
        card.updateSmoothing(small)
        
        // 1. EXPAND: Target is much larger (further from center)
        val large = listOf(
            Point(50.0, 50.0), Point(150.0, 50.0),
            Point(150.0, 150.0), Point(50.0, 150.0)
        )
        // updateGeometric updates card.bounds, but not smoothedBounds
        tracker.updateGeometric(listOf(large))
        card.updateSmoothing(large) // This should use alpha = 0.6 (expansion)
        
        val expandedX = card.smoothedBounds[0].x
        val expandDist = Math.abs(90.0 - expandedX) // Distance moved towards 50.0
        
        // 2. COLLAPSE: Reset and then target is much smaller (closer to center)
        val tracker2 = CardTracker()
        tracker2.updateGeometric(listOf(large))
        val card2 = tracker2.activeCards[0]
        card2.updateSmoothing(large)
        
        tracker2.updateGeometric(listOf(small))
        card2.updateSmoothing(small) // This should use alpha = 0.4 (contraction)
        
        val collapsedX = card2.smoothedBounds[0].x
        val collapseDist = Math.abs(50.0 - collapsedX) // Distance moved towards 90.0
        
        // Verify Bias: Expansion should be snappier (alpha 0.6 vs 0.4)
        assertThat(expandDist).isGreaterThan(collapseDist)
        // 0.6 * 40px = 24px vs 0.4 * 40px = 16px
        assertThat(expandDist).isWithin(1.0).of(24.0)
        assertThat(collapseDist).isWithin(1.0).of(16.0)
    }

    @Test
    fun tracker_expiresAfterTimeout() {
        val tracker = CardTracker()
        
        // Initial detection
        tracker.updateGeometric(listOf(boundsA))
        
        // Simulate time passing (EXPIRY_MS is 600)
        Thread.sleep(700)
        
        val tracked = tracker.updateGeometric(emptyList())
        assertThat(tracked).isEmpty()
    }
}
