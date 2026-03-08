package com.guywithburrito.setfinder.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.guywithburrito.setfinder.R

@Composable
fun WelcomeScreen(onStartClicked: () -> Unit, onSettingsClicked: () -> Unit) {
    Surface(color = MaterialTheme.colors.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.ic_launcher),
                contentDescription = "App Logo",
                modifier = Modifier
                    .size(120.dp)
                    .clip(CircleShape)
            )
            
            Spacer(modifier = Modifier.height(24.dp))
            
            Text(
                text = "SET Finder",
                style = MaterialTheme.typography.h2,
                fontWeight = FontWeight.Black,
                color = MaterialTheme.colors.primary,
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "Identify real-life SETs in seconds using your camera.",
                style = MaterialTheme.typography.subtitle1,
                color = MaterialTheme.colors.onBackground,
                textAlign = TextAlign.Center
            )
            
            Spacer(modifier = Modifier.height(48.dp))
            
            Button(
                onClick = onStartClicked,
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.medium
            ) {
                Text(
                    text = "START SCANNING",
                    modifier = Modifier.padding(8.dp),
                    fontWeight = FontWeight.Bold
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            OutlinedButton(
                onClick = onSettingsClicked,
                modifier = Modifier.fillMaxWidth(),
                shape = MaterialTheme.shapes.medium,
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = MaterialTheme.colors.primary
                )
            ) {
                Text(
                    text = "SETTINGS",
                    modifier = Modifier.padding(8.dp)
                )
            }
        }
    }
}
