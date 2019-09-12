package tk.gustavo.pages;

import org.fluentlenium.core.FluentDriver;
import org.fluentlenium.core.FluentPage;
import org.fluentlenium.core.annotation.Page;
import org.fluentlenium.core.domain.FluentWebElement;
import org.openqa.selenium.By;
import org.openqa.selenium.support.FindBy;

public class BasePage extends FluentPage {


    @Page
    protected HomePage homePage;

    @Page
    protected AboutPage aboutPage;


    @FindBy(linkText = "About")
    protected FluentWebElement aboutButton;

    /**
     * Seeks a Flash Alert from Flask in the page
     *@return true if the WebDriver is able to find an element with the informed class, which means the login has failed.
     */
    public boolean hasFlashAlert(String xPath){
        return el(By.xpath(xPath)).present();
    }

    public boolean isLoggedIn() {
        return el(By.linkText("Logout")).present();
    }

    public boolean hasInvalidFeedback(){
        return el(By.xpath("//div[@class='invalid-feedback']")).present();
    }

    public BasePage logout(){
        if(isLoggedIn()) {
            $(By.linkText("Logout")).click();
        }
        return this;
    }
    
}
