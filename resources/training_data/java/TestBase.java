package tk.gustavo.tests.testng;

import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Optional;
import org.testng.annotations.Parameters;
import tk.gustavo.config.TLDriverFactory;

import java.net.MalformedURLException;

public class TestBase {

    protected WebDriverWait wait;


    @BeforeMethod
    @Parameters(value = {"browser"})
    public void setUpTest(@Optional String browser) throws MalformedURLException {
        TLDriverFactory.setTlDriver(browser);
        wait = new WebDriverWait(TLDriverFactory.getTLDriver(), 15);
    }

    @AfterMethod
    public synchronized void tearDown() throws Exception {
        try {
            TLDriverFactory.getTLDriver().quit();
        } catch (Exception e){

        }
    }

    protected synchronized void executeJavaScript(String script){
        ((JavascriptExecutor) TLDriverFactory.getTLDriver()).executeScript(
                script
        );
    }


}
