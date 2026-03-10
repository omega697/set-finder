package com.guywithburrito.setfinder

import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.ChipExtractor
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.*
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f

@RunWith(RobolectricTestRunner::class)
class SetDetectorJVMTest {

    @Test
    fun detectSets_orchestratesFullPipeline() {
        val mockFinder: CardFinder = mock()
        val mockExtractor: ChipExtractor = mock()
        val mockIdentifier: CardIdentifier = mock()
        val mockProcessor: FrameProcessor = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        
        val detector = SetDetector(mockFinder, mockExtractor, mockIdentifier, mockProcessor)
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        // 1. Mock Detection
        val mockQuad: MatOfPoint2f = mock()
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mockQuad))
        
        // 2. Mock Extraction
        val mockBitmap: Bitmap = mock()
        whenever(mockExtractor.extract(eq(frame), any())).thenReturn(mockBitmap)
        
        // 3. Mock Identification
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        whenever(mockIdentifier.identifyCard(any())).thenReturn(card)
        
        // Execute
        detector.detectSets(frame)
        
        // Verify Interactions
        verify(mockFinder).findLikelyCards(any())
        verify(mockExtractor).extract(eq(frame), any())
        verify(mockIdentifier).identifyCard(any())
    }

    @Test
    fun findSets_correctlyIdentifiesThreeCards() {
        val detector = SetDetector(mock(), mock(), mock(), mock())
        
        val c1 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        val c2 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.TWO, SetCard.Color.RED)
        val c3 = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.THREE, SetCard.Color.RED)
        
        val sets = detector.findSets(listOf(c1, c2, c3))
        assertThat(sets).hasSize(1)
        assertThat(sets[0]).containsExactly(c1, c2, c3)
    }
}
