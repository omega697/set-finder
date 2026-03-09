package com.guywithburrito.setfinder

import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.cv.CardFinder
import com.guywithburrito.setfinder.cv.CardUnwarper
import com.guywithburrito.setfinder.cv.FrameProcessor
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.mockito.kotlin.*
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

@RunWith(RobolectricTestRunner::class)
class SetDetectorJVMTest {

    @Test
    fun detectSets_orchestratesFullPipeline() {
        val mockFinder: CardFinder = mock()
        val mockUnwarper: CardUnwarper = mock()
        val mockIdentifier: CardIdentifier = mock()
        val mockProcessor: FrameProcessor = mock()
        
        whenever(mockProcessor.createMat()).thenReturn(mock())
        
        // Spy on detector to mock internal identifyQuads call (which uses native Mat logic)
        val detector = spy(SetDetector(mockFinder, mockUnwarper, mockIdentifier, mockProcessor))
        
        val frame: Mat = mock()
        whenever(frame.cols()).thenReturn(1000)
        whenever(frame.rows()).thenReturn(1000)
        
        // 1. Mock Detection
        val mockQuad: MatOfPoint2f = mock()
        whenever(mockFinder.findLikelyCards(any())).thenReturn(listOf(mockQuad))
        
        // 2. Mock Identification (Directly mock the internal call to avoid native code)
        val card = SetCard(SetCard.Shape.OVAL, SetCard.Pattern.SOLID, SetCard.Count.ONE, SetCard.Color.RED)
        doReturn(listOf(card)).whenever(detector).identifyQuads(any(), any(), any())
        
        // 3. Execute
        detector.detectSets(frame)
        
        // Verify Interactions
        verify(mockFinder).findLikelyCards(any())
        verify(detector).identifyQuads(any(), any(), any())
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
