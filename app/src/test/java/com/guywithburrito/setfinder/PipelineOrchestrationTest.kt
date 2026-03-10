package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.tracking.CardTracker
import com.guywithburrito.setfinder.tracking.SettingsManager
import com.guywithburrito.setfinder.tracking.TrackedCard
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.*
import org.opencv.core.Mat
import androidx.test.core.app.ApplicationProvider
import com.guywithburrito.setfinder.card.SetCard
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.runTest
import org.opencv.core.Point

@RunWith(RobolectricTestRunner::class)
class PipelineOrchestrationTest {

    @Test
    fun cardDetector_orchestratesDetectionToIdentification() = runTest {
        val appContext = ApplicationProvider.getApplicationContext<android.content.Context>()
        
        val mockFinder: QuadFinder = mock()
        val mockIdentifier: CardIdentifier = mock()
        val mockProcessor: FrameProcessor = mock()
        val mockTracker: CardTracker = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        
        val detector = CardDetector(
            context = appContext,
            scope = this,
            finder = mockFinder,
            identifier = mockIdentifier,
            frameProcessor = mockProcessor,
            tracker = mockTracker
        )
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        // 1. Mock Detection output
        val mockQuad: List<Point> = listOf(Point(0.0, 0.0), Point(1.0, 0.0), Point(1.0, 1.0), Point(0.0, 1.0))
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mock()))
        
        // 2. Mock Tracking output
        val mockTracked: TrackedCard = mock()
        whenever(mockTracker.updateGeometric(any())).thenReturn(listOf(mockTracked))
        whenever(mockTracker.getCardsForIdentification()).thenReturn(emptyList()) // Skip async ID for this simple check
        
        detector.processFrame(frame)
        
        // Verify orchestration
        verify(mockProcessor).resize(any(), any(), any(), any(), any(), any())
        verify(mockFinder).findLikelyCards(any())
        verify(mockTracker).updateGeometric(any())
    }
}
