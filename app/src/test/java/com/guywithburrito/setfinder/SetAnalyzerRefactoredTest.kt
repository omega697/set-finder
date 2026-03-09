package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.tracking.CardTracker
import com.guywithburrito.setfinder.tracking.SettingsManager
import com.guywithburrito.setfinder.tracking.TrackedCard
import com.guywithburrito.setfinder.card.SetSolver
import com.guywithburrito.setfinder.ml.CardIdentifier
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.*
import org.opencv.core.Mat
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.MainScope

@RunWith(RobolectricTestRunner::class)
class SetAnalyzerRefactoredTest {

    @Test
    fun analyzeMat_orchestratesIdentificationAndTracking() {
        val appContext = ApplicationProvider.getApplicationContext<android.content.Context>()
        val settingsManager = SettingsManager(appContext)
        
        val mockFinder: CardFinder = mock()
        val mockTracker: CardTracker = mock()
        val mockDetector: SetDetector = mock()
        val mockSolver: SetSolver = mock()
        val mockProcessor: FrameProcessor = mock()
        val mockIdentifier: CardIdentifier = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        
        val analyzer = SetAnalyzer(
            appContext, 
            MainScope(), 
            settingsManager,
            finder = mockFinder,
            tracker = mockTracker,
            solver = mockSolver,
            frameProcessor = mockProcessor,
            identifier = mockIdentifier,
            detector = mockDetector
        )
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        val mockTracked: TrackedCard = mock()
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mock()))
        whenever(mockTracker.updateGeometric(any())).thenReturn(listOf(mockTracked))
        whenever(mockSolver.solve(any())).thenReturn(emptyList())
        
        analyzer.analyzeMat(frame)
        
        // Verify high-level orchestration through interfaces
        verify(mockProcessor).resize(any(), any(), any(), any(), any(), any())
        verify(mockFinder).findLikelyCards(any())
        verify(mockTracker).updateGeometric(any())
        verify(mockSolver).solve(any())
    }
}
