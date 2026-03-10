package com.guywithburrito.setfinder

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.QuadFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
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
import org.opencv.core.MatOfPoint2f
import androidx.test.core.app.ApplicationProvider
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.MainScope

/**
 * This JVM-based unit test suite evaluates the orchestration logic across the 
 * entire pipeline. It ensures that data flows correctly between the high-level 
 * Analyzer, the modular Detector, and the underlying CV/ML components, 
 * verifying the "plumbing" of the system using mocks.
 */
@RunWith(RobolectricTestRunner::class)
class PipelineOrchestrationTest {

    @Test
    fun setAnalyzer_orchestratesIdentificationAndTracking() {
        val appContext = ApplicationProvider.getApplicationContext<android.content.Context>()
        val settingsManager = SettingsManager(appContext)
        
        val mockTracker: CardTracker = mock()
        val mockFinder: QuadFinder = mock()
        val mockProcessor: FrameProcessor = mock()
        val mockIdentifier: CardIdentifier = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        
        val analyzer = SetAnalyzer(
            appContext, 
            MainScope(), 
            settingsManager,
            tracker = mockTracker,
            frameProcessor = mockProcessor,
            finder = mockFinder,
            identifier = mockIdentifier
        )
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        val mockTracked: TrackedCard = mock()
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mock()))
        whenever(mockTracker.updateGeometric(any())).thenReturn(listOf(mockTracked))
        
        analyzer.analyzeMat(frame)
        
        // Verify high-level orchestration
        verify(mockProcessor).resize(any(), any(), any(), any(), any(), any())
        verify(mockFinder).findLikelyCards(any())
        verify(mockTracker).updateGeometric(any())
    }

    @Test
    fun setDetector_orchestratesDetectionToIdentification() {
        val mockFinder: QuadFinder = mock()
        val mockExtractor: ChipExtractor = mock()
        val mockIdentifier: CardIdentifier = mock()
        val mockProcessor: FrameProcessor = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        whenever(mockProcessor.createMatOfPoint2f(any())).thenReturn(mock())
        
        val detector = SetDetector(mockFinder, mockExtractor, mockIdentifier, mockProcessor)
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        // Mock Detection output
        val mockQuad: MatOfPoint2f = mock()
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mockQuad))
        
        // Mock Extraction output
        val mockBitmap: Bitmap = mock()
        whenever(mockExtractor.extract(eq(frame), any())).thenReturn(mockBitmap)
        
        // Mock Identification output
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        whenever(mockIdentifier.identifyCard(any())).thenReturn(card)
        
        // Execute the detector's entry point
        val detected = detector.detectCards(frame)
        
        // Verify the handoff between components
        verify(mockFinder).findLikelyCards(any())
        verify(mockExtractor).extract(eq(frame), any())
        verify(mockIdentifier).identifyCard(any())
        
        // Verify results contain the detected card data
        assertThat(detected).hasSize(1)
        assertThat(detected[0].card).isEqualTo(card)
    }
}
