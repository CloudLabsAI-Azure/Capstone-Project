# Challenge Lab

# Capstone Project 

### Overall Estimated Duration: 4 Hours

## Overview

In this challenge, you will work with a Chainlit-based conversational application that utilizes Dapr’s publish-subscribe (pub/sub) messaging model to orchestrate customer service escalations through intelligent AI agents. The solution seamlessly integrates several Azure services—including Azure OpenAI for natural language processing, Cosmos DB for data persistence, and Azure Service Bus for reliable messaging. When AI agents reach their resolution limits, the system intelligently escalates the case to a human support agent by triggering a Logic Apps workflow that sends an approval request via email. This hands-on lab offers valuable insight into how AI-driven conversational interfaces, event-driven architecture, and workflow automation can be combined to create responsive, scalable, and human-aware customer support systems.

## Objective 

By the end of this lab, you will be able to:

- **Multi-Agent Business Persona Workflow**: Experience a collaborative workflow involving three distinct personas—Software Engineer, Product Owner, and User. The Software Engineer writes and submits code, the Product Owner reviews and approves the changes, and upon approval, the solution is automatically pushed to GitHub for deployment and version control.

## Prerequisites

Participants should have:

- Basic understanding of Azure services such as Azure OpenAI, and models.
- Experience with deploying applications using Azure Developer CLI (AZD).
- Exposure to conversational AI tools like Streamlit or similar frameworks.

## Explanation of Components

- **Azure OpenAI**: A cloud-based service that provides access to advanced language models, enabling natural language processing, content generation, and conversational AI capabilities. It allows you to integrate powerful AI-driven features into your applications securely and at scale.
- **Azure Container Apps**: A serverless container hosting service that allows you to deploy and scale microservices and containerized applications without managing infrastructure.

## Getting Started with the lab

Welcome to your Azure Agentic AI Workshop, Let's begin by making the most of this experience:

## Accessing Your Lab Environment

Once you're ready to dive in, your virtual machine and **Guide** will be right at your fingertips within your web browser.

![Access Your VM and Lab Guide](./media/VmImage.png)

## Lab Guide Zoom In/Zoom Out

To adjust the zoom level for the environment page, click the **A↕ : 100%** icon located next to the timer in the lab environment.

![](./media/agg2.png)

## Virtual Machine & Lab Guide

Your virtual machine is your workhorse throughout the workshop. The lab guide is your roadmap to success.

## Exploring Your Lab Resources

To get a better understanding of your lab resources and credentials, navigate to the **Environment** tab.

![Explore Lab Resources](./media/agg3.png)

## Utilizing the Split Window Feature

For convenience, you can open the lab guide in a separate window by selecting the **Split Window** button from the Top right corner.

![Use the Split Window Feature](./media/agg4.png)

## Managing Your Virtual Machine

Feel free to **Start, Stop, or Restart (2)** your virtual machine as needed from the **Resources (1)** tab. Your experience is in your hands!

![Manage Your Virtual Machine](./media/agg5.png)

<!-- ## Lab Duration Extension

1. To extend the duration of the lab, kindly click the **Hourglass** icon in the top right corner of the lab environment.

    ![Manage Your Virtual Machine](./media/media/gext.png)

    >**Note:** You will get the **Hourglass** icon when 10 minutes are remaining in the lab.

2. Click **OK** to extend your lab duration.

   ![Manage Your Virtual Machine](./media/media/gext2.png)

3. If you have not extended the duration prior to when the lab is about to end, a pop-up will appear, giving you the option to extend. Click **OK** to proceed. -->

> **Note:** Please ensure the script continues to run and is not terminated after accessing the environment.

## Let's Get Started with Azure Portal

1. On your virtual machine, click on the Azure Portal icon.
2. You'll see the **Sign into Microsoft Azure** tab. Here, enter your credentials:

   - **Email/Username:** <inject key="AzureAdUserEmail"></inject>

     ![Enter Your Username](./media/gt-5.png)

3. Next, provide your password:

   - **Password:** <inject key="AzureAdUserPassword"></inject>

     ![Enter Your Password](./media/gt-4.png)

4. If **Action required** pop-up window appears, click on **Ask later**.
5. If prompted to **stay signed in**, you can click **No**.
6. If a **Welcome to Microsoft Azure** pop-up window appears, simply click **"Cancel"** to skip the tour.

## Steps to sign in to the github.

1. Open a new browser tab and navigate to the following URL:  **[https://github.com/login](https://github.com/login)**

   ![Enter Your Password](./media/200.png)

1. On the **Sign in to GitHub** page, enter your personal Github credentials in **Username (1)** and **Password (2)** then click on **Sign in (3)**.

   ![Enter Your Password](./media/201.png)

1. If prompted for a verification code, enter the **GitHub verification code** received in your registered email or authentication app.

## Steps to Proceed with MFA Setup if "Ask Later" Option is Not Visible

1. At the **"More information required"** prompt, select **Next**.

1. On the **"Keep your account secure"** page, select **Next** twice.

1. **Note:** If you don’t have the Microsoft Authenticator app installed on your mobile device:

   - Open **Google Play Store** (Android) or **App Store** (iOS).
   - Search for **Microsoft Authenticator** and tap **Install**.
   - Open the **Microsoft Authenticator** app, select **Add account**, then choose **Work or school account**.

1. A **QR code** will be displayed on your computer screen.

1. In the Authenticator app, select **Scan a QR code** and scan the code displayed on your screen.

1. After scanning, click **Next** to proceed.

1. On your phone, enter the number shown on your computer screen in the Authenticator app and select **Next**.
1. If prompted to stay signed in, you can click "No."

1. If a **Welcome to Microsoft Azure** pop-up window appears, simply click "Maybe Later" to skip the tour.

## Support Contact

The CloudLabs support team is available 24/7, 365 days a year, via email and live chat to ensure seamless assistance at any time. We offer dedicated support channels tailored specifically for both learners and instructors, ensuring that all your needs are promptly and efficiently addressed.

Learner Support Contacts:

- Email Support: [cloudlabs-support@spektrasystems.com](mailto:cloudlabs-support@spektrasystems.com)
- Live Chat Support: https://cloudlabs.ai/labs-support

Click **Next** from the bottom right corner to embark on your Lab journey!

![Start Your Azure Journey](./media/agg6.png)

Now you're all set to explore the powerful world of technology. Feel free to reach out if you have any questions along the way. Enjoy your workshop!
